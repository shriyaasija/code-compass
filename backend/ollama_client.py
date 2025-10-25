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
            model_names = [m.get("name", "").split(":") for m in models]
            
            if self.model not in model_names:
                print(f"Model '{self.model}' not found. Available: {model_names}")
                print(f"   Download with: ollama pull {self.model}")
            else:
                print(f"Connected to Ollama - Using model: {self.model}")
                
        except requests.exceptions.RequestException as e:
            print(f"Ollama not running! Start with: ollama serve")
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
                timeout=120  # 2 min timeout for large generations
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "").strip()
            
        except requests.exceptions.Timeout:
            return "Generation timed out. Try a simpler query or smaller model."
        except requests.exceptions.RequestException as e:
            return f"Ollama error: {str(e)}"
    
    def chat(self, 
             messages: List[Dict[str, str]], 
             temperature: float = 0.3,
             max_tokens: int = 2000) -> str:
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        try:
            response = requests.post(
                self.chat_url,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("message", {}).get("content", "").strip()
            
        except requests.exceptions.Timeout:
            return "Chat timed out. Try a simpler query."
        except requests.exceptions.RequestException as e:
            return f"Chat error: {str(e)}"

def test_ollama():
    """Quick test to verify Ollama is working"""
    print("\n" + "="*70)
    print("TESTING OLLAMA CONNECTION")
    print("="*70 + "\n")
    
    try:
        llm = OllamaLLM(model="mistral")
        
        # Test basic generation
        prompt = "Explain what a Python decorator is in one sentence."
        print(f"Prompt: {prompt}\n")
        
        response = llm.generate(prompt, temperature=0.3)
        print(f"Response: {response}\n")
        
        # Test chat
        messages = [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": "What does 'API' stand for?"}
        ]
        
        print("Testing chat mode...")
        chat_response = llm.chat(messages)
        print(f"Response: {chat_response}\n")
        
        print("Ollama tests passed!\n")
        return True
        
    except Exception as e:
        print(f"Test failed: {e}\n")
        return False


if __name__ == "__main__":
    test_ollama()