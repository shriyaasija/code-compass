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
        
    def get_function_code(self, file_path, start_line=None, end_line=None):
        lines = self.get_file_content(file_path)

        if not lines:
            return f"# Could not read file: {file_path}"
        
        if start_line is None or end_line is None:
            return ''.join(lines)
            
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
    
    def __init__(self, repo_path: str, llm_provider: str = "ollama", llm_model: str = "qwen3:8b"):
        from pathlib import Path
        self.repo_path = Path(repo_path).resolve()
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        
        if not self.repo_path.exists():
            raise FileNotFoundError(f"Repository path does not exist: {self.repo_path}")
        
        if not self.repo_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {self.repo_path}")
        
        print(f"✅ Chatbot initialized for repo: {self.repo_path} (provider: {llm_provider})")
        
        self.retriever = CodeRetriever(str(self.repo_path))

    def generate_response(self, user_query: str, filtered_functions: list) -> str:
        """
        Generate response with full context - NO truncation.
        Uses proper prompting for long-context understanding.
        """
        if not filtered_functions:
            return "No relevant code found for your query."
        
        # Build FULL context - no limits
        context_parts = []
        
        for idx, func in enumerate(filtered_functions, 1):
            try:
                start_line = func.get('start_line')
                end_line = func.get('end_line')
                
                code = self.retriever.get_function_code(
                    func.get('path', func.get('file_path')), 
                    start_line, 
                    end_line
                )
                
                context_parts.append(
                    f"\n{'='*60}\n"
                    f"[CODE BLOCK {idx}/{len(filtered_functions)}]\n"
                    f"File: {func.get('path', func.get('file_path'))}\n"
                    f"Name: {func.get('name', func.get('title', 'Unknown'))}\n"
                    f"Lines: {start_line}-{end_line}\n"
                    f"Score: {func.get('_score', func.get('score', 'N/A'))}\n"
                    f"{'='*60}\n"
                    f"{code}\n"
                )
            except Exception as e:
                print(f"⚠️ Failed to read {func['file_path']}: {e}")
                continue
        
        context = "\n".join(context_parts)
        
        # Always use streaming for stability with large contexts
        print(f"📡 Streaming {len(filtered_functions)} functions (full context, provider: {self.llm_provider})...")
        
        if self.llm_provider == "lmstudio":
            response, tokens_used = self._call_lmstudio_streaming(user_query, context, len(filtered_functions))
        else:
            response, tokens_used = self._call_ollama_streaming(user_query, context, len(filtered_functions))
        
        if not response or response.startswith("Error") or response.startswith("Cannot"):
            print(f"⚠️ LLM returned: {response}")
            return (response if response else "Failed to generate response."), {"total_tokens": 0}
        
        return response, tokens_used

    def _call_ollama_streaming(self, user_query: str, context: str, num_functions: int) -> tuple:
        """
        Streaming with NEEDLE-IN-HAYSTACK prompting for long contexts.
        This is the modern approach for handling massive contexts.
        """
        import requests
        import json
        
        # Modern long-context prompting technique
        system_prompt = """You are a precise code analysis expert with perfect recall of long contexts.

Your capabilities:
- You can process and remember large amounts of code
- You find exact details across the entire context
- You answer the SPECIFIC question asked, not general summaries
- You cite exact file names and line numbers when referencing code

Response format:
1. Direct answer to the question
2. Relevant code references with file paths
3. Technical explanation of how/why it works"""

        # Put question at BOTH start and end (needle-in-haystack technique)
        full_prompt = f"""<QUESTION>
{user_query}
</QUESTION>

Below are {num_functions} code blocks from the repository. Read ALL of them carefully.

<CODE_CONTEXT>
{context}
</CODE_CONTEXT>

Now answer this question based on the code above:
<QUESTION>
{user_query}
</QUESTION>

Provide a detailed, technical answer that directly addresses the question. Reference specific code blocks by their file paths."""
        
        try:
            print("🔄 Sending full context to Ollama...")
            print(f"   Context size: ~{len(context)//1000}K chars")
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "qwen3:8b",
                    "prompt": system_prompt + "\n\n" + full_prompt,
                    "stream": True,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 4000,
                        "num_ctx": 32768,
                        "top_p": 0.95,
                        "repeat_penalty": 1.1
                    }
                },
                stream=True,
                timeout=900
            )
            
            if response.status_code != 200:
                print(f"❌ Ollama returned status {response.status_code}")
                return f"Ollama API error: {response.status_code}", {"total_tokens": 0}
            
            full_response = ""
            chunk_count = 0
            token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        full_response += chunk.get("response", "")
                        chunk_count += 1
                        
                        # Ollama sends final token counts when done=true
                        if chunk.get("done"):
                            token_usage["prompt_tokens"] = chunk.get("prompt_eval_count", 0)
                            token_usage["completion_tokens"] = chunk.get("eval_count", 0)
                            token_usage["total_tokens"] = token_usage["prompt_tokens"] + token_usage["completion_tokens"]
                        
                        if chunk_count % 50 == 0:
                            print(".", end="", flush=True)
                            
                    except json.JSONDecodeError:
                        continue
            
            print(f"\n✅ Generated {len(full_response)} chars from {chunk_count} chunks")
            
            if not full_response.strip():
                print("⚠️ Empty response - context might be too large for model")
                return "The context was too large. Try increasing the threshold to get fewer functions.", {"total_tokens": 0}
            
            return full_response.strip(), token_usage
            
        except requests.exceptions.ConnectionError:
            return "Cannot connect to Ollama. Make sure it's running: ollama serve", {"total_tokens": 0}
        except requests.exceptions.Timeout:
            return "Generation timed out. The context may be extremely large.", {"total_tokens": 0}
        except Exception as e:
            print(f"❌ Streaming error: {str(e)}")
            return f"Error calling Ollama: {str(e)}", {"total_tokens": 0}

    def _call_lmstudio_streaming(self, user_query: str, context: str, num_functions: int) -> tuple:
        """
        Streaming with LM Studio's OpenAI-compatible API.
        """
        import requests
        import json
        
        system_prompt = """You are a precise code analysis expert with perfect recall of long contexts.

Your capabilities:
- You can process and remember large amounts of code
- You find exact details across the entire context
- You answer the SPECIFIC question asked, not general summaries
- You cite exact file names and line numbers when referencing code

Response format:
1. Direct answer to the question
2. Relevant code references with file paths
3. Technical explanation of how/why it works"""

        user_prompt = f"""<QUESTION>
{user_query}
</QUESTION>

Below are {num_functions} code blocks from the repository. Read ALL of them carefully.

<CODE_CONTEXT>
{context}
</CODE_CONTEXT>

Now answer this question based on the code above:
<QUESTION>
{user_query}
</QUESTION>

Provide a detailed, technical answer that directly addresses the question. Reference specific code blocks by their file paths."""
        
        try:
            print("🔄 Sending full context to LM Studio...")
            print(f"   Context size: ~{len(context)//1000}K chars")
            
            response = requests.post(
                "http://localhost:1234/v1/chat/completions",
                json={
                    "model": self.llm_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 4000,
                    "stream": True
                },
                stream=True,
                timeout=900
            )
            
            if response.status_code != 200:
                print(f"❌ LM Studio returned status {response.status_code}")
                return f"LM Studio API error: {response.status_code}", {"total_tokens": 0}
            
            full_response = ""
            chunk_count = 0
            
            # Since LM Studio doesn't always reliably include token usage in stream completion, 
            # we'll approximate based on character counts for prompt and generation
            prompt_chars = len(system_prompt) + len(user_prompt)
            prompt_tokens_est = prompt_chars // 4
            
            for line in response.iter_lines():
                if line:
                    line_str = line.decode("utf-8") if isinstance(line, bytes) else line
                    if line_str.startswith("data: "):
                        data_str = line_str[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            full_response += delta.get("content", "")
                            chunk_count += 1
                            
                            if chunk_count % 50 == 0:
                                print(".", end="", flush=True)
                                
                        except json.JSONDecodeError:
                            continue
            
            print(f"\n✅ Generated {len(full_response)} chars from {chunk_count} chunks")
            
            if not full_response.strip():
                print("⚠️ Empty response - context might be too large for model")
                return "The context was too large. Try increasing the threshold to get fewer functions.", {"total_tokens": 0}
            
            completion_tokens_est = len(full_response) // 4
            token_usage = {
                "prompt_tokens": prompt_tokens_est,
                "completion_tokens": completion_tokens_est,
                "total_tokens": prompt_tokens_est + completion_tokens_est
            }
            
            return full_response.strip(), token_usage
            
        except requests.exceptions.ConnectionError:
            return "Cannot connect to LM Studio. Make sure the server is running.", {"total_tokens": 0}
        except requests.exceptions.Timeout:
            return "Generation timed out. The context may be extremely large.", {"total_tokens": 0}
        except Exception as e:
            print(f"❌ Streaming error: {str(e)}")
            return f"Error calling LM Studio: {str(e)}", {"total_tokens": 0}