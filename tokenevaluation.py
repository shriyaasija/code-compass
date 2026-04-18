#!/usr/bin/env python3
"""
Code Compass Pipeline Evaluator
================================
Evaluates the full pipeline (clone → tree-sitter parse → LLM summarize → MCTS search → query)
for given repos and queries, then compares token usage against Claude baseline.

Usage:
    LLM_PROVIDER=lmstudio python evaluate_pipeline.py
    LLM_PROVIDER=ollama python evaluate_pipeline.py
"""

import os
import sys
import json
import time
import subprocess
import tempfile
import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# ── Baseline from manual Claude runs ──────────────────────────────────────────
# Format: (repo_url, query, claude_tokens_used)
EVAL_REPOS = [
    {
        "repo_url": "https://github.com/kennethreitz/records",
        "repo_id":  "kennethreitz_records",
        "queries": [
            ("How does the records library handle transactions?",          120_000),
            ("Where can I find the license information for the records library?", 127_500),
            ("How does the records library handle exceptions?",            137_500),
        ],
    },
    {
        "repo_url": "https://github.com/keleshev/schema",
        "repo_id":  "keleshev_schema",
        "queries": [
            ("What strategies does the schema library employ to handle complex nested schemas?", 160_000),
            ("What mechanisms are used to ensure API keys are validated before processing requests?", 167_500),
            ("How does the library support inheritance and optional fields?",  183_500),
        ],
    },
    {
        "repo_url": "https://github.com/docopt/docopt",
        "repo_id":  "docopt_docopt",
        "queries": [
            ("How does docopt handle Unicode strings in command line arguments?", 110_000),
            ("How does the formal_usage() function contribute to the documentation of docopt?", 132_500),
            ("Where can I find examples of how to use docopt?",           142_500),
        ],
    },
]


# ── Result containers ──────────────────────────────────────────────────────────
@dataclass
class QueryResult:
    query: str
    claude_tokens: int
    our_tokens: int = 0
    response_length: int = 0
    latency_s: float = 0.0
    error: Optional[str] = None

    @property
    def token_ratio(self) -> float:
        if self.claude_tokens == 0:
            return 0.0
        return self.our_tokens / self.claude_tokens

    @property
    def token_savings(self) -> int:
        return self.claude_tokens - self.our_tokens


@dataclass
class RepoResult:
    repo_url: str
    repo_id: str
    summarizer_tokens: int = 0
    summarizer_llm_calls: int = 0
    summarizer_nodes: int = 0
    summarizer_time_s: float = 0.0
    query_results: List[QueryResult] = field(default_factory=list)
    clone_time_s: float = 0.0
    parse_time_s: float = 0.0
    error: Optional[str] = None

    @property
    def total_our_tokens(self) -> int:
        return self.summarizer_tokens + sum(q.our_tokens for q in self.query_results)

    @property
    def total_claude_tokens(self) -> int:
        return sum(q.claude_tokens for q in self.query_results)

    @property
    def cumulative_savings(self) -> int:
        return self.total_claude_tokens - self.total_our_tokens

    @property
    def avg_token_ratio(self) -> float:
        if self.total_claude_tokens == 0:
            return 0.0
        return self.total_our_tokens / self.total_claude_tokens


# ── Pipeline runner ────────────────────────────────────────────────────────────
class PipelineEvaluator:
    def __init__(self, llm_provider: str = "lmstudio", llm_model: str = None):
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.llm_client = None
        self.api_base = "http://localhost:8000"
        self.results: List[RepoResult] = []

    # ── LLM setup ──────────────────────────────────────────────────────────
    def _init_llm(self):
        """Initialize the LLM client directly (no API server needed)."""
        sys.path.insert(0, str(Path(__file__).parent))
        if self.llm_provider == "lmstudio":
            from backend.lmstudio_client import LMStudioLLM
            kwargs = {}
            if self.llm_model:
                kwargs["model"] = self.llm_model
            self.llm_client = LMStudioLLM(**kwargs)
        else:
            from backend.ollama_client import OllamaLLM
            model = self.llm_model or "qwen3:8b"
            self.llm_client = OllamaLLM(model=model)
        print(f"✅ LLM client ready: {self.llm_provider} / {self.llm_client.model}")

    # ── Clone ──────────────────────────────────────────────────────────────
    def _clone_repo(self, repo_url: str, target_dir: str) -> float:
        t0 = time.time()
        print(f"   📥 Cloning {repo_url}…")
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, target_dir],
            check=True, capture_output=True, timeout=120
        )
        elapsed = time.time() - t0
        print(f"   ✅ Cloned in {elapsed:.1f}s")
        return elapsed

    # ── Build + summarize tree ─────────────────────────────────────────────
    def _build_and_summarize(self, repo_path: str, repo_id: str, cache_dir: Path):
        from backend.tree_builder import build_directory_tree
        from backend.code_parser import CodeParser, enrich_tree_with_code_structure
        from backend.summarizer import TreeSummarizer

        json_path = str(cache_dir / f"{repo_id}_pageindex.json")

        if Path(json_path).exists():
            print(f"   ♻️  Using cached PageIndex: {json_path}")
            return json_path, 0, 0, 0, 0.0

        # Stage 1 – directory tree
        print("   🌳 Building directory tree…")
        t0 = time.time()
        tree = build_directory_tree(repo_path)

        # Stage 2 – tree-sitter enrichment
        print("   🧩 Enriching with tree-sitter…")
        parser = CodeParser()
        tree = enrich_tree_with_code_structure(tree, parser)
        parse_time = time.time() - t0

        # Stage 3 – bottom-up LLM summarisation
        print("   📝 Summarising (bottom-up)… this may take a while")
        t1 = time.time()
        summarizer = TreeSummarizer(self.llm_client, verbose=True)
        tree_dict = tree.to_dict()
        tree_dict = summarizer.summarize_tree(tree_dict, repo_path)
        summarize_time = time.time() - t1

        # Persist
        with open(json_path, "w") as f:
            json.dump(tree_dict, f, indent=2)
        print(f"   ✅ PageIndex saved → {json_path}")

        return (
            json_path,
            summarizer.total_tokens,
            summarizer.llm_calls,
            summarizer.summary_count,
            summarize_time,
        )

    # ── Query via MCTS ─────────────────────────────────────────────────────
    def _run_query(self, repo_id: str, repo_path: str, json_path: str,
                   query: str, tree_search, chatbots: dict) -> QueryResult:
        from backend.retrieval import ProductionChatbot

        if repo_id not in chatbots:
            chatbots[repo_id] = ProductionChatbot(
                repo_path,
                llm_provider=self.llm_provider,
                llm_model=self.llm_client.model
            )

        t0 = time.time()
        try:
            # MCTS search
            filtered = tree_search.search_and_format_for_chatbot(
                repo_id=repo_id, query=query, top_k=5
            )
            if not filtered:
                return QueryResult(query=query, claude_tokens=0, error="No results from MCTS")

            # LLM response generation
            chatbot = chatbots[repo_id]
            response, tokens_used = chatbot.generate_response(
                user_query=query, filtered_functions=filtered
            )
            latency = time.time() - t0

            our_tokens = tokens_used.get("total_tokens", 0) if tokens_used else 0
            return QueryResult(
                query=query,
                claude_tokens=0,   # filled in later
                our_tokens=our_tokens,
                response_length=len(response),
                latency_s=latency,
            )
        except Exception as e:
            latency = time.time() - t0
            return QueryResult(query=query, claude_tokens=0, error=str(e), latency_s=latency)

    # ── Main evaluation loop ───────────────────────────────────────────────
    def run(self):
        self._init_llm()

        from backend.code_index2 import MCTSTreeSearch

        cache_dir = Path("eval_cache")
        cache_dir.mkdir(exist_ok=True)

        tree_search = MCTSTreeSearch(self.llm_client, threshold=0.5)
        chatbots: dict = {}

        for repo_cfg in EVAL_REPOS:
            repo_url = repo_cfg["repo_url"]
            repo_id  = repo_cfg["repo_id"]
            queries  = repo_cfg["queries"]

            print(f"\n{'='*70}")
            print(f"🏗️  EVALUATING: {repo_url}")
            print(f"{'='*70}")

            result = RepoResult(repo_url=repo_url, repo_id=repo_id)
            self.results.append(result)

            try:
                # Clone
                with tempfile.TemporaryDirectory() as tmpdir:
                    clone_dir = os.path.join(tmpdir, repo_id)
                    result.clone_time_s = self._clone_repo(repo_url, clone_dir)

                    # Build + summarize
                    (json_path,
                     sum_tokens, sum_calls, sum_nodes, sum_time) = \
                        self._build_and_summarize(clone_dir, repo_id, cache_dir)

                    result.summarizer_tokens   = sum_tokens
                    result.summarizer_llm_calls = sum_calls
                    result.summarizer_nodes    = sum_nodes
                    result.summarizer_time_s   = sum_time
                    result.parse_time_s        = result.clone_time_s  # reuse slot

                    # Load tree into MCTS search
                    tree_search.load_repository_tree(repo_id, json_path)

                    # Run queries
                    for (query_text, claude_tokens) in queries:
                        print(f"\n   ❓ Query: {query_text[:70]}…")
                        qr = self._run_query(
                            repo_id, clone_dir, json_path,
                            query_text, tree_search, chatbots
                        )
                        qr.query         = query_text
                        qr.claude_tokens = claude_tokens
                        result.query_results.append(qr)
                        status = f"❌ {qr.error}" if qr.error else f"✅ {qr.our_tokens:,} tokens, {qr.latency_s:.1f}s"
                        print(f"   {status}")

            except Exception as e:
                result.error = str(e)
                print(f"\n❌ Repo-level error: {e}")

        self._generate_report()

    # ── Report generation ──────────────────────────────────────────────────
    def _generate_report(self):
        ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path("eval_results")
        out.mkdir(exist_ok=True)

        # ── JSON raw results ────────────────────────────────────────────────
        raw = []
        for r in self.results:
            raw.append({
                "repo_url": r.repo_url,
                "repo_id":  r.repo_id,
                "error":    r.error,
                "summarizer": {
                    "tokens":    r.summarizer_tokens,
                    "llm_calls": r.summarizer_llm_calls,
                    "nodes":     r.summarizer_nodes,
                    "time_s":    round(r.summarizer_time_s, 1),
                },
                "queries": [
                    {
                        "query":        q.query,
                        "claude_tokens":q.claude_tokens,
                        "our_tokens":   q.our_tokens,
                        "savings":      q.token_savings,
                        "ratio":        round(q.token_ratio, 4),
                        "latency_s":    round(q.latency_s, 2),
                        "error":        q.error,
                    }
                    for q in r.query_results
                ],
                "totals": {
                    "our_tokens":    r.total_our_tokens,
                    "claude_tokens": r.total_claude_tokens,
                    "savings":       r.cumulative_savings,
                    "ratio":         round(r.avg_token_ratio, 4),
                },
            })

        json_path = out / f"eval_{ts}.json"
        with open(json_path, "w") as f:
            json.dump(raw, f, indent=2)
        print(f"\n📄 Raw JSON saved → {json_path}")

        # ── Markdown report ─────────────────────────────────────────────────
        md_path = out / f"eval_{ts}.md"
        lines   = self._build_markdown(ts)
        with open(md_path, "w") as f:
            f.write("\n".join(lines))
        print(f"📊 Markdown report → {md_path}")

        # ── Console summary ─────────────────────────────────────────────────
        print("\n" + "\n".join(self._build_markdown(ts, console=True)))

    def _build_markdown(self, ts: str, console: bool = False) -> List[str]:
        lines = [
            "# Code Compass Pipeline Evaluation",
            f"**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"**LLM Provider:** {self.llm_provider}",
            f"**Model:** {self.llm_client.model if self.llm_client else 'N/A'}",
            "",
            "---",
            "",
            "## Summary Table",
            "",
            "| Repository | Summarizer Tokens | Summarizer Calls | "
            "Total Our Tokens | Total Claude Tokens | Savings | Ratio |",
            "|---|---|---|---|---|---|---|",
        ]

        for r in self.results:
            lines.append(
                f"| [{r.repo_id}]({r.repo_url}) "
                f"| {r.summarizer_tokens:,} "
                f"| {r.summarizer_llm_calls:,} "
                f"| {r.total_our_tokens:,} "
                f"| {r.total_claude_tokens:,} "
                f"| {r.cumulative_savings:,} "
                f"| {r.avg_token_ratio:.2%} |"
            )

        lines += ["", "---", "", "## Per-Repo Breakdown", ""]

        for r in self.results:
            lines += [
                f"### {r.repo_id}",
                f"**URL:** {r.repo_url}",
            ]
            if r.error:
                lines.append(f"**⚠️ Error:** {r.error}")
                lines.append("")
                continue

            lines += [
                "",
                "#### Summarizer Phase",
                "",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Tokens used | {r.summarizer_tokens:,} |",
                f"| LLM calls   | {r.summarizer_llm_calls:,} |",
                f"| Nodes summarised | {r.summarizer_nodes:,} |",
                f"| Time (s)    | {r.summarizer_time_s:.1f} |",
                "",
                "#### Query Results",
                "",
                "| # | Query (truncated) | Our Tokens | Claude Tokens | Savings | Ratio | Latency(s) | Status |",
                "|---|---|---|---|---|---|---|---|",
            ]

            for i, q in enumerate(r.query_results, 1):
                truncated = (q.query[:55] + "…") if len(q.query) > 55 else q.query
                status    = "❌ " + (q.error[:30] if q.error else "") if q.error else "✅"
                lines.append(
                    f"| {i} | {truncated} "
                    f"| {q.our_tokens:,} "
                    f"| {q.claude_tokens:,} "
                    f"| {q.token_savings:,} "
                    f"| {q.token_ratio:.2%} "
                    f"| {q.latency_s:.1f} "
                    f"| {status} |"
                )

            lines += [
                "",
                f"**Cumulative Our Tokens** (summarizer + queries): **{r.total_our_tokens:,}**",
                f"**Total Claude Tokens** (queries only baseline): **{r.total_claude_tokens:,}**",
                f"**Token Savings**: **{r.cumulative_savings:,}** ({r.avg_token_ratio:.2%} of Claude usage)",
                "",
            ]

        # ── Grand totals ────────────────────────────────────────────────────
        grand_our    = sum(r.total_our_tokens    for r in self.results)
        grand_claude = sum(r.total_claude_tokens for r in self.results)
        grand_save   = grand_claude - grand_our
        grand_ratio  = (grand_our / grand_claude) if grand_claude else 0

        lines += [
            "---",
            "## Grand Total",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Our Tokens (all repos) | {grand_our:,} |",
            f"| Total Claude Tokens baseline  | {grand_claude:,} |",
            f"| Total Savings                 | {grand_save:,} |",
            f"| Overall Ratio (ours/Claude)   | {grand_ratio:.2%} |",
            "",
        ]

        return lines


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    provider = os.environ.get("LLM_PROVIDER", "lmstudio").strip().lower()
    model    = os.environ.get("LLM_MODEL", None)

    print("=" * 70)
    print("🧭 CODE COMPASS PIPELINE EVALUATOR")
    print("=" * 70)
    print(f"  LLM Provider : {provider}")
    print(f"  LLM Model    : {model or '(auto-detect)'}")
    print(f"  Repos        : {len(EVAL_REPOS)}")
    print(f"  Queries/repo : {len(EVAL_REPOS[0]['queries'])}")
    print("=" * 70)

    evaluator = PipelineEvaluator(llm_provider=provider, llm_model=model)
    evaluator.run()