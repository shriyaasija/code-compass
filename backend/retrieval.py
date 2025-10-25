from pathlib import Path
from functools import lru_cache
import tempfile
import subprocess

class CodeRetriever:

    def __init__(self, repo_path):
        self.repo_path = Path(repo_path)
        self._file_cache = {}

    @lru_cache(maxsize=500)
    def get_file_content(self, file_path):
        full_path = self.repo_path / file_path

        try:
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.readlines()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return []
        
    def get_function_code(self, file_path, start_line, end_line):
        lines = self.get_file_content(file_path)

        if not lines:
            return f"# Could not read file: {file_path}"
        
        code = ''.join(lines[start_line-1:end_line])
        return code
    
    def get_multiple_functions(self, function_list):
        results = []
        
        for func in function_list:
            code = self.get_function_code(
                func['file_path'],
                func['start_line'],
                func['end_line']
            )

            result = func.copy()
            result['code'] = code
            results.append(result)
        
        return results

class ProductionChatbot:
    
    def __init__(self, repo_path: str):
        self.retriever = CodeRetriever(repo_path)
        print(f"✅ Chatbot initialized for repo: {repo_path}")
    
    def generate_response(self, user_query, filtered_functions):
        if not filtered_functions:
            return "I couldn't find any relevant code for your question."
        
        # Step 1: Get actual code for all filtered functions
        functions_with_code = self.retriever.get_multiple_functions(filtered_functions)
        
        if not functions_with_code:
            return "Failed to retrieve code for the relevant functions."
        
        # Step 2: Format the response
        response = self._format_response(user_query, functions_with_code)
        
        return response
    
    def _format_response(self, query, functions_with_code):
        response_parts = [
            f"# Answer to: {query}\n",
            f"Found {len(functions_with_code)} relevant functions:\n"
        ]
        
        for i, func in enumerate(functions_with_code, 1):
            response_parts.append(
                f"\n## {i}. {func['name']}\n\n"
                f"**File:** `{func['file_path']}`  \n"
                f"**Lines:** {func['start_line']}-{func['end_line']}  \n"
            )
            
            if func.get('docstring'):
                response_parts.append(f"**Description:** {func['docstring']}  \n")
            
            response_parts.append(f"\n```python\n{func['code']}\n```\n")
        
        return '\n'.join(response_parts)
    
    def generate_llm_context(self, user_query, filtered_functions):
        functions_with_code = self.retriever.get_multiple_functions(filtered_functions)
        
        if not functions_with_code:
            return None
        
        # Build context for LLM
        context_parts = []
        
        for func in functions_with_code:
            context_parts.append(
                f"Function: {func['name']}\n"
                f"File: {func['file_path']}\n"
                f"Signature: {func.get('signature', 'N/A')}\n"
            )
            
            if func.get('docstring'):
                context_parts.append(f"Description: {func['docstring']}\n")
            
            context_parts.append(f"Code:\n{func['code']}\n")
        
        context = "\n" + "="*60 + "\n".join(context_parts)
        
        # This is what you'd send to your LLM
        llm_prompt = f"""User Question: {user_query}

Relevant Code from Repository:
{context}

Based on the code above, answer the user's question in a clear and helpful way."""
        
        return llm_prompt

