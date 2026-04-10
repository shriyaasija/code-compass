import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any


class TreeSummarizer:
    """
    Bottom-up LLM summarizer for code trees.
    
    Traverses the tree depth-first, summarizing leaves first,
    then combining child summaries to produce parent summaries.
    """
    
    def __init__(self, llm_client, max_code_chars: int = 2000, verbose: bool = True):
        """
        Args:
            llm_client: LMStudioLLM or OllamaLLM instance with .chat() method.
            max_code_chars: Max characters of source code to include in prompt.
            verbose: Print progress messages.
        """
        self.llm = llm_client
        self.max_code_chars = max_code_chars
        self.verbose = verbose
        self.summary_count = 0
        self.llm_calls = 0
    
    def summarize_tree(self, tree: Dict, repo_path: str) -> Dict:
        """
        Generate summaries for every node in the tree, bottom-up.
        Modifies the tree in-place and returns it.
        
        Args:
            tree: Tree dict (from tree_builder.to_dict())
            repo_path: Path to the cloned repository (for reading source code)
        """
        self.summary_count = 0
        self.llm_calls = 0
        start = time.time()
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"📝 BOTTOM-UP TREE SUMMARIZATION")
            print(f"{'='*70}")
        
        self._summarize_node(tree, repo_path)
        
        elapsed = time.time() - start
        if self.verbose:
            print(f"\n✅ Summarized {self.summary_count} nodes in {elapsed:.1f}s "
                  f"({self.llm_calls} LLM calls)")
        
        return tree
    
    def _summarize_node(self, node: Dict, repo_path: str) -> str:
        """
        Recursively summarize a node. Children are summarized first.
        Returns the summary string for this node.
        """
        children = node.get('nodes', node.get('children', []))
        node_type = node.get('type', node.get('node_type', ''))
        title = node.get('title', node.get('name', 'unknown'))
        
        # RECURSIVE CASE: summarize children first (bottom-up)
        child_summaries = []
        for child in children:
            child_summary = self._summarize_node(child, repo_path)
            child_title = child.get('title', child.get('name', '?'))
            child_type = child.get('type', child.get('node_type', '?'))
            if child_summary:
                child_summaries.append({
                    'title': child_title,
                    'type': child_type,
                    'summary': child_summary,
                })
        
        # Generate summary based on node type
        if node_type in ('function', 'method'):
            summary = self._summarize_code_node(node, repo_path)
        elif node_type == 'class':
            summary = self._summarize_class_node(node, child_summaries)
        elif node_type.startswith('file'):
            summary = self._summarize_file_node(node, child_summaries, repo_path)
        elif node_type == 'folder':
            summary = self._summarize_folder_node(node, child_summaries)
        elif node_type == 'repository':
            summary = self._summarize_repo_node(node, child_summaries)
        else:
            # Unknown type — use title as summary
            summary = title
        
        # Store summary on node
        node['summary'] = summary
        self.summary_count += 1
        
        if self.verbose and self.summary_count % 20 == 0:
            print(f"   Summarized {self.summary_count} nodes...")
        
        return summary
    
    def _summarize_code_node(self, node: Dict, repo_path: str) -> str:
        """Summarize a function or method by reading its source code."""
        file_path = node.get('path', node.get('file_path', ''))
        start_line = node.get('start_line')
        end_line = node.get('end_line')
        title = node.get('title', 'unknown')
        
        # Try to read the actual source code
        source_code = ""
        if file_path and start_line is not None and end_line is not None:
            full_path = os.path.join(repo_path, file_path) if not os.path.isabs(file_path) else file_path
            try:
                with open(full_path, 'r', errors='ignore') as f:
                    lines = f.readlines()
                    source_code = ''.join(lines[start_line:end_line + 1])
                    # Truncate if too long
                    if len(source_code) > self.max_code_chars:
                        source_code = source_code[:self.max_code_chars] + "\n... (truncated)"
            except (FileNotFoundError, IOError):
                source_code = ""
        
        if not source_code:
            # Fallback: use title and any existing docstring
            return node.get('summary', f"Function {title}")
        
        prompt = f"""Summarize this Python function in 1-2 sentences. Focus on WHAT it does, not HOW.

```python
{source_code}
```

Respond with ONLY the summary, no extra text."""
        
        return self._call_llm(prompt, fallback=f"Function {title}")
    
    def _summarize_class_node(self, node: Dict, child_summaries: List[Dict]) -> str:
        """Summarize a class from its method summaries."""
        title = node.get('title', 'unknown')
        
        if not child_summaries:
            return f"Class {title}"
        
        methods_text = "\n".join([
            f"  - {cs['title']}: {cs['summary']}" 
            for cs in child_summaries[:15]
        ])
        if len(child_summaries) > 15:
            methods_text += f"\n  ... and {len(child_summaries) - 15} more methods"
        
        prompt = f"""Summarize this class in 1-2 sentences based on its methods.

Class: {title}
Methods:
{methods_text}

Respond with ONLY the summary."""
        
        return self._call_llm(prompt, fallback=f"Class {title} with {len(child_summaries)} methods")
    
    def _summarize_file_node(self, node: Dict, child_summaries: List[Dict],
                             repo_path: str) -> str:
        """Summarize a file from its function/class summaries."""
        title = node.get('title', node.get('name', 'unknown'))
        
        if not child_summaries:
            # For files with no parsed children (non-code files, etc.)
            return f"File {title}"
        
        children_text = "\n".join([
            f"  - {cs['title']} ({cs['type']}): {cs['summary']}"
            for cs in child_summaries[:20]
        ])
        if len(child_summaries) > 20:
            children_text += f"\n  ... and {len(child_summaries) - 20} more items"
        
        prompt = f"""Summarize this source file in 1-2 sentences based on its contents.

File: {title}
Contents:
{children_text}

Respond with ONLY the summary."""
        
        return self._call_llm(prompt, fallback=f"File {title} with {len(child_summaries)} items")
    
    def _summarize_folder_node(self, node: Dict, child_summaries: List[Dict]) -> str:
        """Summarize a folder from its children summaries."""
        title = node.get('title', node.get('name', 'unknown'))
        
        if not child_summaries:
            return f"Folder {title}"
        
        children_text = "\n".join([
            f"  - {cs['title']} ({cs['type']}): {cs['summary']}"
            for cs in child_summaries[:15]
        ])
        if len(child_summaries) > 15:
            children_text += f"\n  ... and {len(child_summaries) - 15} more items"
        
        prompt = f"""Summarize this folder/module in 1-2 sentences based on its contents.

Folder: {title}/
Contents:
{children_text}

Respond with ONLY the summary."""
        
        return self._call_llm(prompt, fallback=f"Module {title} with {len(child_summaries)} items")
    
    def _summarize_repo_node(self, node: Dict, child_summaries: List[Dict]) -> str:
        """Summarize the root repository node."""
        title = node.get('title', node.get('name', 'unknown'))
        
        if not child_summaries:
            return f"Repository {title}"
        
        children_text = "\n".join([
            f"  - {cs['title']} ({cs['type']}): {cs['summary']}"
            for cs in child_summaries[:20]
        ])
        
        prompt = f"""Summarize this code repository in 2-3 sentences.

Repository: {title}
Top-level contents:
{children_text}

Respond with ONLY the summary."""
        
        return self._call_llm(prompt, fallback=f"Repository {title}")
    
    def _call_llm(self, prompt: str, fallback: str = "") -> str:
        """Call the LLM and return the response, with fallback on failure."""
        self.llm_calls += 1
        try:
            messages = [
                {"role": "system", "content": "You are a code documentation assistant. "
                 "Provide concise, factual summaries. No markdown, no bullet points, "
                 "just plain English sentences."},
                {"role": "user", "content": prompt}
            ]
            response = self.llm.chat(messages, temperature=0.1, max_tokens=150)
            if response and response.strip():
                # Clean up: remove quotes, extra whitespace
                summary = response.strip().strip('"').strip("'").strip()
                # Limit length
                if len(summary) > 300:
                    summary = summary[:297] + "..."
                return summary
        except Exception as e:
            if self.verbose:
                print(f"   ⚠️ LLM call failed: {e}")
        return fallback