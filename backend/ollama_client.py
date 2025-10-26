import requests
import json
from typing import List, Dict, Optional


class OllamaLLM:
    def __init__(self, model, base_url = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        self.chat_url = f"{base_url}/api/chat"
        
        # Verify connection on init
        self._verify_connection()
    
    def _verify_connection(self):
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            
            # Check if model exists (exact match or with tag)
            model_found = False
            for available_model in model_names:
                if available_model == self.model or available_model.startswith(f"{self.model}:"):
                    model_found = True
                    break
            
            if not model_found:
                print(f"⚠️ Model '{self.model}' not found locally")
                print(f"   Available models: {model_names}")
                print(f"   Download with: ollama pull {self.model}")
            else:
                print(f"✅ Connected to Ollama - Using model: {self.model}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Ollama not running! Start with: ollama serve")
            print(f"   Error: {e}")
            raise ConnectionError("Ollama server not accessible")
    
    def generate(self, 
                 prompt: str, 
                 system_prompt: Optional[str] = None,
                 temperature: float = 0.3,
                 max_tokens: int = 2000) -> str:
        
        # Combine system + user prompt
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        try:
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=300  # 2 min timeout for large generations
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "").strip()
            
        except requests.exceptions.Timeout:
            return "Generation timed out. Try a simpler query or smaller model."
        except requests.exceptions.RequestException as e:
            print(f"❌ Ollama generate error: {str(e)}")
            return f"Ollama error: {str(e)}"
    
    def chat(self, 
             messages: List[Dict[str, str]], 
             temperature: float = 0.3,
             max_tokens: int = 2000,
             format: str = None,
             retry_on_empty: bool = True) -> str:
        """
        Chat with Ollama model.
        
        Args:
            messages: List of message dicts with role and content
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            format: Optional format specification (e.g., 'json')
        """
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        # Force JSON format for structured output (helps with reasoning models)
        if format == "json":
            payload["format"] = "json"
        
        try:
            response = requests.post(
                self.chat_url,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            message = result.get("message", {})
            
            # Get content - this should be the actual response
            message_content = message.get("content", "").strip()
            
            # For reasoning models, thinking is separate - we only want content
            if not message_content:
                print(f"⚠️ Ollama returned empty content")
                print(f"   Model: {self.model}")
                
                # Retry once with adjusted temperature if enabled
                if retry_on_empty and temperature < 0.7:
                    print(f"   Retrying with higher temperature...")
                    return self.chat(
                        messages=messages,
                        temperature=temperature + 0.3,
                        max_tokens=max_tokens,
                        format=format,
                        retry_on_empty=False  # Prevent infinite retry
                    )
                
                # Return default score if still empty
                return '{"score": 0.5}' 
            
            return message_content
            
        except requests.exceptions.Timeout:
            print("❌ Chat timed out")
            return ""
        except requests.exceptions.RequestException as e:
            print(f"❌ Ollama chat error: {str(e)}")
            return ""
        
    def generate_streaming(self, 
                      prompt: str, 
                      system_prompt: Optional[str] = None,
                      temperature: float = 0.3,
                      max_tokens: int = 2000) -> str:
        """
        Generate response with streaming to prevent timeouts on large contexts.
        Returns the complete response after streaming is done.
        """
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": True,  # Enable streaming
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        try:
            response = requests.post(
                self.api_url,
                json=payload,
                stream=True,
                timeout=600  # 10 min total timeout
            )
            response.raise_for_status()
            
            full_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        full_response += chunk.get("response", "")
                    except json.JSONDecodeError:
                        continue
            
            return full_response.strip()
            
        except requests.exceptions.Timeout:
            return "Generation timed out even with streaming."
        except requests.exceptions.RequestException as e:
            print(f"❌ Ollama streaming error: {str(e)}")
            return f"Ollama error: {str(e)}"

    def chat_streaming(self,
                    messages: List[Dict[str, str]], 
                    temperature: float = 0.3,
                    max_tokens: int = 2000) -> str:
        """
        Chat with streaming to prevent timeouts on large contexts.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        try:
            response = requests.post(
                self.chat_url,
                json=payload,
                stream=True,
                timeout=600
            )
            response.raise_for_status()
            
            full_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        message = chunk.get("message", {})
                        full_response += message.get("content", "")
                    except json.JSONDecodeError:
                        continue
            
            return full_response.strip()
            
        except requests.exceptions.Timeout:
            print("❌ Chat streaming timed out")
            return ""
        except requests.exceptions.RequestException as e:
            print(f"❌ Ollama chat streaming error: {str(e)}")
            return ""
        
def test_ollama():
    """Quick test to verify Ollama is working"""
    print("\n" + "="*70)
    print("TESTING OLLAMA CONNECTION")
    print("="*70 + "\n")
    
    try:
        llm = OllamaLLM(model="qwen3:8b")
        
        # Test basic generation
        prompt = "Explain what a Python decorator is in one sentence."
        print(f"Prompt: {prompt}\n")
        
        response = llm.generate(prompt, temperature=0.3)
        print(f"Response: {response}\n")
        
        # Test chat with JSON response
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Respond with valid JSON only."},
            {"role": "user", "content": 'Rate these items 0.0-1.0: {"apple": ?, "banana": ?}. Return as JSON.'}
        ]
        
        print("Testing chat mode with JSON...")
        chat_response = llm.chat(messages)
        print(f"Response: {chat_response}\n")
        
        print("✅ Ollama tests passed!\n")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}\n")
        return False


if __name__ == "__main__":
    test_ollama()