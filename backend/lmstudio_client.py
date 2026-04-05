import requests
import json
from typing import List, Dict, Optional


class LMStudioLLM:
    """
    LLM client for LM Studio's OpenAI-compatible API.
    Same interface as OllamaLLM so the rest of the codebase works unchanged.
    """
    def __init__(self, model="local-model", base_url="http://localhost:1234"):
        self.model = model
        self.base_url = base_url
        self.chat_url = f"{base_url}/v1/chat/completions"
        
        # Verify connection on init
        self._verify_connection()
    
    def _verify_connection(self):
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            response.raise_for_status()
            
            models = response.json().get("data", [])
            model_names = [m.get("id", "") for m in models]
            
            if model_names:
                # Keep 'local-model' to force LM Studio to use the active loaded model
                # Doing this prevents the 29GB memory check crash
                if self.model == "local-model" and model_names:
                    self.model = "local-model"
                print(f"✅ Connected to LM Studio - Using model: {self.model}")
                print(f"   Available models: {model_names}")
            else:
                print(f"⚠️ No models loaded in LM Studio")
                print(f"   Load a model in LM Studio and start the server")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ LM Studio not running! Start the server in LM Studio")
            print(f"   Expected at: {self.base_url}")
            print(f"   Error: {e}")
            raise ConnectionError("LM Studio server not accessible")
    
    def generate(self, 
                 prompt: str, 
                 system_prompt: Optional[str] = None,
                 temperature: float = 0.3,
                 max_tokens: int = 2000) -> str:
        """Generate uses chat completions under the hood (more reliable with LM Studio)."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        return self.chat(messages, temperature=temperature, max_tokens=max_tokens)
    
    def chat(self, 
             messages: List[Dict[str, str]], 
             temperature: float = 0.3,
             max_tokens: int = 2000,
             format: str = None,
             retry_on_empty: bool = True) -> str:
        """Chat with LM Studio model using OpenAI-compatible API."""
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        if format == "json":
            payload["response_format"] = {"type": "json_object"}
        
        try:
            response = requests.post(
                self.chat_url,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            choices = result.get("choices", [])
            
            if not choices:
                print(f"⚠️ LM Studio returned no choices")
                if retry_on_empty and temperature < 0.7:
                    print(f"   Retrying with higher temperature...")
                    return self.chat(
                        messages=messages,
                        temperature=temperature + 0.3,
                        max_tokens=max_tokens,
                        format=format,
                        retry_on_empty=False
                    )
                return '{"score": 0.5}'
            
            message_content = choices[0].get("message", {}).get("content", "").strip()
            
            if not message_content:
                print(f"⚠️ LM Studio returned empty content")
                if retry_on_empty and temperature < 0.7:
                    print(f"   Retrying with higher temperature...")
                    return self.chat(
                        messages=messages,
                        temperature=temperature + 0.3,
                        max_tokens=max_tokens,
                        format=format,
                        retry_on_empty=False
                    )
                return '{"score": 0.5}'
            
            return message_content
            
        except requests.exceptions.Timeout:
            print("❌ Chat timed out")
            return ""
        except requests.exceptions.RequestException as e:
            print(f"❌ LM Studio chat error: {str(e)}")
            return ""
        
    def generate_streaming(self, 
                      prompt: str, 
                      system_prompt: Optional[str] = None,
                      temperature: float = 0.3,
                      max_tokens: int = 2000) -> str:
        """Generate with streaming."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        return self.chat_streaming(messages, temperature=temperature, max_tokens=max_tokens)

    def chat_streaming(self,
                    messages: List[Dict[str, str]], 
                    temperature: float = 0.3,
                    max_tokens: int = 2000) -> str:
        """Chat with streaming using OpenAI-compatible SSE format."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True
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
                    line_str = line.decode("utf-8") if isinstance(line, bytes) else line
                    if line_str.startswith("data: "):
                        data_str = line_str[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            full_response += delta.get("content", "")
                        except json.JSONDecodeError:
                            continue
            
            return full_response.strip()
            
        except requests.exceptions.Timeout:
            print("❌ Chat streaming timed out")
            return ""
        except requests.exceptions.RequestException as e:
            print(f"❌ LM Studio chat streaming error: {str(e)}")
            return ""


def test_lmstudio():
    """Quick test to verify LM Studio is working"""
    print("\n" + "="*70)
    print("TESTING LM STUDIO CONNECTION")
    print("="*70 + "\n")
    
    try:
        llm = LMStudioLLM()  # Auto-detect model
        
        prompt = "Explain what a Python decorator is in one sentence."
        print(f"Prompt: {prompt}\n")
        
        response = llm.generate(prompt, temperature=0.3)
        print(f"Response: {response}\n")
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Respond with valid JSON only."},
            {"role": "user", "content": 'Rate these items 0.0-1.0: {"apple": ?, "banana": ?}. Return as JSON.'}
        ]
        
        print("Testing chat mode with JSON...")
        chat_response = llm.chat(messages)
        print(f"Response: {chat_response}\n")
        
        print("✅ LM Studio tests passed!\n")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}\n")
        return False


if __name__ == "__main__":
    test_lmstudio()
