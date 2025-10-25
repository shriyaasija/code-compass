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
        # Convert to Path object and resolve to absolute path
        from pathlib import Path
        self.repo_path = Path(repo_path).resolve()
        
        # Validate path exists
        if not self.repo_path.exists():
            raise FileNotFoundError(f"Repository path does not exist: {self.repo_path}")
        
        if not self.repo_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {self.repo_path}")
        
        print(f"✅ Chatbot initialized for repo: {self.repo_path}")
        
        # Initialize retriever with validated path
        self.retriever = CodeRetriever(str(self.repo_path))

    def generate_response(self, user_query: str, filtered_functions: list):
        if not filtered_functions:
            return "No functions provided to analyze."
        
        # Get code for all functions
        functions_with_code = self.retriever.get_multiple_functions(filtered_functions)
        
        # Build context for Ollama
        context = self._build_context(functions_with_code)
        
        # Generate AI response using Ollama
        response = self._call_ollama(user_query, context)
        
        return response

    def _build_context(self, functions_with_code: list) -> str:
        context_parts = []
        
        for i, func in enumerate(functions_with_code, 1):
            context_parts.append(f"""
    ### Function {i}: {func['name']}
    File: {func['file_path']}
    Lines: {func['start_line']}-{func['end_line']}
    {func['code']}
    """)
    
        return "\n".join(context_parts)

    def _call_ollama(self, user_query: str, context: str) -> str:
        import requests
        
        system_prompt = """You are an expert software engineer helping developers understand code.

    Your task:
    1. Answer the user's question clearly and concisely
    2. Reference specific code snippets when relevant
    3. Explain HOW things work, not just WHAT they are
    4. Be direct and practical

    Guidelines:
    - Focus on the question asked
    - Use code formatting when referencing code
    - Keep explanations beginner-friendly but accurate"""

        full_prompt = f"""User Question: {user_query}

    Relevant Code:
    {context}

    Based on the code above, provide a clear answer to the user's question."""

        try:
            # Call Ollama API
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "qwen3:8b",
                    "prompt": system_prompt + "\n\n" + full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 1500
                    }
                },
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                return f"Ollama API error: {response.status_code}"
                
        except requests.exceptions.ConnectionError:
            return "Cannot connect to Ollama. Make sure it's running: ollama serve"
        except requests.exceptions.Timeout:
            return "Ollama response timed out. Try a simpler query."
        except Exception as e:
            return f"Error calling Ollama: {str(e)}"
    
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

