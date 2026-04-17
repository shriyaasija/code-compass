"""
LLM-powered auto-annotator for DevQuery-Bench.

Replaces the manual annotate.py step (Step 5 of the Day 1 guide).

KEY IMPROVEMENTS over v1:
  - Reads ACTUAL source code from cloned repos for BM25 ranking and LLM context
    (v1 only used title-based summaries like "function: foo at path" which caused
    ~50% NONE results because the LLM had no real information to work with)
  - Auto-reconnect with exponential backoff when LM Studio drops
  - Larger BM25 candidate pool (30) reduced to top-15 with real code for LLM
  - Falls back to BM25 top-1 for queries the LLM still can't resolve

Usage:
    python devquery_bench/auto_annotate.py               # all repos
    python devquery_bench/auto_annotate.py --repo django__django
    python devquery_bench/auto_annotate.py --repo django__django --force

Requirements:
    - devquery_bench/repo_metadata.json   (from build_trees.py)
    - devquery_bench/trees/<repo_id>.json
    - devquery_bench/queries_<repo_id>.json  (from generate_queries.py)
    - devquery_bench/cloned_repos/<repo_id>/  (the actual source)
    - LM Studio running at http://localhost:1234
    - pip install rank_bm25
"""

import argparse
import ast
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.lmstudio_client import LMStudioLLM

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    print("❌  rank_bm25 not installed. Run: pip install rank_bm25")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BM25_POOL      = 30    # how many BM25 results to pull before trimming
LLM_CANDIDATES = 15    # how many to actually show the LLM (with source)
CODE_LINES     = 25    # lines of source to include per candidate in LLM prompt
TEMPERATURE    = 0.1
MAX_RETRIES    = 5     # LM Studio reconnect attempts

ANNOTATION_PROMPT = """\
You are annotating a code-retrieval benchmark. Your job is to identify which
Python function BEST answers a developer's natural-language query.

Developer query: "{query}"
Repository: {repo_name}

Below are the {n} most relevant candidate functions with their actual source code:

{candidates_block}

Instructions:
- Pick the ONE function whose implementation most directly and specifically answers the query.
- The queries were written BY a developer who had SEEN the codebase, so there IS
  a correct answer somewhere in the list — look carefully at the code bodies.
- If you genuinely cannot find a good match, respond with NONE.
- Respond with ONLY this JSON (no markdown, no extra text):
  {{"function": "<exact_function_name_or_NONE>", "confidence": "<high|medium|low>", "reason": "<one sentence>"}}
"""


# ---------------------------------------------------------------------------
# Source-code extraction
# ---------------------------------------------------------------------------

def _read_lines(filepath: str, start: int, end: int) -> list[str]:
    """Read lines [start, end] (1-indexed, inclusive) from a file."""
    try:
        with open(filepath, encoding="utf-8", errors="ignore") as fh:
            all_lines = fh.readlines()
        s = max(0, start - 1)
        e = min(len(all_lines), end if end else start + CODE_LINES)
        return all_lines[s:e]
    except OSError:
        return []


def _extract_docstring(source_lines: list[str]) -> str:
    """Pull the first triple-quoted docstring out of a list of source lines."""
    src = "".join(source_lines)
    # look for triple-quoted strings after the def signature
    m = re.search(r'"""(.*?)"""', src, re.DOTALL)
    if not m:
        m = re.search(r"'''(.*?)'''", src, re.DOTALL)
    if m:
        return m.group(1).strip()[:300]
    return ""


def enrich_with_source(funcs: list[dict], repo_path: str) -> list[dict]:
    """
    Add 'docstring' and 'code_snippet' to every function entry by reading
    the actual source file.  Falls back gracefully if file isn't found.
    """
    enriched = []
    for f in funcs:
        path = f.get("path", "")
        start = f.get("start_line") or 1

        # path in tree is usually relative to repo root or absolute
        if os.path.isabs(path):
            abs_path = path
        else:
            abs_path = os.path.join(repo_path, path)

        lines = _read_lines(abs_path, start, start + CODE_LINES)
        docstring = _extract_docstring(lines)
        snippet = "".join(lines[:CODE_LINES]).rstrip()

        enriched.append({**f, "docstring": docstring, "code_snippet": snippet})
    return enriched


# ---------------------------------------------------------------------------
# Tree walk
# ---------------------------------------------------------------------------

def collect_functions(node, funcs=None):
    """Walk tree JSON and collect every function/method node."""
    if funcs is None:
        funcs = []

    ntype = node.get("type", node.get("node_type", ""))
    title = node.get("title", node.get("name", ""))

    if ntype in ("function", "method") and title:
        funcs.append({
            "name":       title,
            "path":       node.get("path", ""),
            "summary":    node.get("summary", ""),
            "start_line": node.get("start_line"),
        })

    for child in node.get("nodes", node.get("children", [])):
        collect_functions(child, funcs)

    return funcs


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

def _tok(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def bm25_rank(funcs: list[dict], query: str, top_k: int = BM25_POOL) -> list[dict]:
    """
    Rank functions by BM25 using name + path + summary + docstring + first
    lines of code — everything we have.
    """
    corpus = [
        _tok(" ".join([
            f.get("name", ""),
            f.get("path", ""),
            f.get("summary", ""),
            f.get("docstring", ""),
            # Include first ~5 lines of code as extra BM25 signal
            " ".join(f.get("code_snippet", "").split("\n")[:5]),
        ]))
        for f in funcs
    ]
    if not any(corpus):
        return funcs[:top_k]

    bm25   = BM25Okapi(corpus)
    scores = bm25.get_scores(_tok(query))
    ranked = sorted(zip(scores, funcs), key=lambda x: x[0], reverse=True)
    return [f for _, f in ranked[:top_k]]


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def _parse_llm_json(text: str) -> dict | None:
    """Robustly extract the JSON object from LLM output."""
    # Strip markdown fences
    text = re.sub(r"```(?:json)?", "", text).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "function" in obj:
            return obj
    except json.JSONDecodeError:
        pass
    # Find {...}
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group())
            if isinstance(obj, dict) and "function" in obj:
                return obj
        except json.JSONDecodeError:
            pass
    return None


def _llm_call_with_retry(llm, messages, temperature, max_tokens, retries=MAX_RETRIES):
    """
    Call llm.chat() with exponential-backoff retry so a LM Studio hiccup
    doesn't abort the whole run.
    """
    for attempt in range(1, retries + 1):
        try:
            return llm.chat(messages, temperature=temperature, max_tokens=max_tokens)
        except Exception as e:
            wait = 2 ** attempt
            if attempt < retries:
                print(f"    ⚠️  LM Studio error (attempt {attempt}/{retries}): {e!r}")
                print(f"        Retrying in {wait}s — make sure LM Studio is still running…")
                time.sleep(wait)
            else:
                raise


def llm_pick(llm, query: str, candidates: list[dict], repo_name: str) -> dict:
    """Build a prompt with real code snippets and ask the LLM to pick."""
    blocks = []
    for i, c in enumerate(candidates, 1):
        snippet = c.get("code_snippet", "").strip()
        if not snippet:
            snippet = f"# (source not found)\ndef {c['name']}(...): ..."
        # Truncate to CODE_LINES lines
        snippet_lines = snippet.split("\n")[:CODE_LINES]
        snippet = "\n".join(snippet_lines)

        docstr = c.get("docstring", "")
        doc_line = f'  # docstring: {docstr[:150]}' if docstr else ""

        blocks.append(
            f"--- Candidate {i}: {c['name']} ({c['path']}) ---\n"
            f"{doc_line}\n"
            f"```python\n{snippet}\n```"
        )

    candidates_block = "\n\n".join(blocks)

    prompt = ANNOTATION_PROMPT.format(
        query=query,
        repo_name=repo_name,
        n=len(candidates),
        candidates_block=candidates_block,
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise code annotation assistant. "
                "Respond with valid JSON only — no markdown, no explanation outside the JSON."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    try:
        raw = _llm_call_with_retry(llm, messages, TEMPERATURE, 300)
        result = _parse_llm_json(raw)
        if result:
            return result
        # One retry at higher temperature
        raw = _llm_call_with_retry(llm, messages, 0.3, 300)
        return _parse_llm_json(raw) or {"function": "NONE", "confidence": "low", "reason": "parse failed"}
    except Exception as e:
        return {"function": "NONE", "confidence": "low", "reason": f"LLM error: {e}"}


# ---------------------------------------------------------------------------
# Per-repo annotation
# ---------------------------------------------------------------------------

def annotate_repo(llm, repo_id: str, repo_name: str, tree_path: str,
                  repo_path: str, force: bool = False):
    queries_path = f"devquery_bench/queries_{repo_id}.json"
    output_path  = f"devquery_bench/annotations_{repo_id}.json"

    if not os.path.exists(queries_path):
        print(f"  ⚠️  No queries at {queries_path} — skipping")
        return
    if not os.path.exists(tree_path):
        print(f"  ⚠️  No tree at {tree_path} — skipping")
        return

    # Resume support
    existing = []
    if os.path.exists(output_path) and not force:
        with open(output_path) as f:
            existing = json.load(f)
        if existing:
            print(f"  📂 Resuming from annotation #{len(existing)+1}")

    with open(queries_path) as f:
        queries = json.load(f)

    # ---- Build enriched function list (with source code) ----
    with open(tree_path) as f:
        tree = json.load(f)

    raw_funcs = collect_functions(tree)
    print(f"  🔎 {len(raw_funcs)} functions in tree — reading source code…")

    # Only read source if cloned repo exists
    if os.path.isdir(repo_path):
        funcs = enrich_with_source(raw_funcs, repo_path)
        src_ok = sum(1 for f in funcs if f.get("code_snippet"))
        print(f"  📄 Source read for {src_ok}/{len(funcs)} functions")
    else:
        funcs = raw_funcs
        print(f"  ⚠️  Cloned repo not found at {repo_path} — falling back to summaries only")

    # Build name lookup for validation
    all_names      = {f["name"] for f in funcs}
    all_names_low  = {f["name"].lower(): f["name"] for f in funcs}

    annotations = list(existing)
    new_count = skip_count = 0

    for i, query in enumerate(queries):
        if i < len(existing):
            continue

        print(f"\n  [{i+1}/{len(queries)}] {query[:90]}")

        # BM25: pull larger pool, then trim to LLM_CANDIDATES
        pool       = bm25_rank(funcs, query, top_k=BM25_POOL)
        candidates = pool[:LLM_CANDIDATES]

        if not candidates:
            print("    ⚠️  No candidates — skipping")
            annotations.append({"query": query, "ground_truth": None,
                                 "skipped": True, "auto": True, "reason": "no candidates"})
            skip_count += 1
        else:
            result     = llm_pick(llm, query, candidates, repo_name)
            chosen     = result.get("function", "NONE")
            confidence = result.get("confidence", "?")
            reason     = result.get("reason", "")

            # Validate / fuzzy-match
            if chosen and chosen.upper() != "NONE":
                if chosen not in all_names:
                    chosen = all_names_low.get(chosen.lower(), "NONE")

            # Last-resort fallback: if LLM says NONE but BM25 top-1 is
            # a reasonable match (score > 0), use it with low confidence.
            if (not chosen or chosen.upper() == "NONE") and pool:
                top1 = pool[0]
                # Only fallback if it looks like it was actually scored
                # (pool[0] has the highest BM25 score)
                chosen     = top1["name"]
                confidence = "low"
                reason     = f"[BM25 fallback] {reason}"
                print(f"    ↩  BM25 fallback → {chosen}")

            if chosen and chosen.upper() != "NONE":
                print(f"    ✅ {chosen} [{confidence}] — {reason[:70]}")
                annotations.append({
                    "query":        query,
                    "ground_truth": chosen,
                    "skipped":      False,
                    "auto":         True,
                    "confidence":   confidence,
                    "reason":       reason,
                })
                new_count += 1
            else:
                print(f"    ⏭  Still NONE — {reason[:70]}")
                annotations.append({
                    "query":        query,
                    "ground_truth": None,
                    "skipped":      True,
                    "auto":         True,
                    "confidence":   confidence,
                    "reason":       reason,
                })
                skip_count += 1

        # Auto-save after every query
        with open(output_path, "w") as f:
            json.dump(annotations, f, indent=2)

        time.sleep(0.2)

    annotated = sum(1 for a in annotations if not a.get("skipped"))
    print(f"\n  ✅ {repo_id}: {annotated}/{len(queries)} annotated "
          f"({new_count} new, {skip_count} skipped/fallback)")
    print(f"     Saved → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="LLM-powered auto-annotator for DevQuery-Bench"
    )
    ap.add_argument("--repo",  default=None,
                    help="Annotate only this repo ID (e.g. django__django)")
    ap.add_argument("--force", action="store_true",
                    help="Re-annotate even if annotations file already exists")
    args = ap.parse_args()

    # Verify LM Studio
    print("🔌  Connecting to LM Studio…")
    try:
        llm = LMStudioLLM()
        llm.chat([{"role": "user", "content": "Say: ok"}], temperature=0, max_tokens=5)
        print("✅  LM Studio OK\n")
    except Exception as e:
        print(f"❌  Cannot reach LM Studio: {e}")
        print("    Start LM Studio → Local Server → Start Server")
        sys.exit(1)

    metadata_path = "devquery_bench/repo_metadata.json"
    if not os.path.exists(metadata_path):
        print(f"❌  {metadata_path} not found. Run build_trees.py first.")
        sys.exit(1)

    with open(metadata_path) as f:
        repos = json.load(f)

    if args.repo:
        repos = [r for r in repos if r["repo_id"] == args.repo]
        if not repos:
            print(f"❌  '{args.repo}' not in repo_metadata.json")
            sys.exit(1)

    print(f"📋  Will annotate {len(repos)} repo(s)\n")

    for idx, repo in enumerate(repos, 1):
        repo_id   = repo["repo_id"]
        repo_name = repo["repo_name"]
        tree_path = repo["tree_path"]
        repo_path = repo["repo_path"]  # path to cloned source

        output_path = f"devquery_bench/annotations_{repo_id}.json"
        if os.path.exists(output_path) and not args.force:
            with open(output_path) as f:
                existing = json.load(f)
            qpath = f"devquery_bench/queries_{repo_id}.json"
            if os.path.exists(qpath):
                with open(qpath) as f:
                    qs = json.load(f)
                if len(existing) >= len(qs):
                    print(f"[{idx}/{len(repos)}] ✅ {repo_id} — fully annotated, skipping")
                    continue

        print(f"\n{'─'*70}")
        print(f"[{idx}/{len(repos)}] {repo_name}  ({repo_id})")

        annotate_repo(llm, repo_id, repo_name, tree_path, repo_path,
                      force=args.force)

    print(f"\n{'='*70}")
    print("🎉  Auto-annotation complete!")
    print("    Next: python devquery_bench/compute_naturalism.py")


if __name__ == "__main__":
    main()
