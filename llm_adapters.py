"""
LLM Integration Module

Provides adapters for different LLM APIs to test various models
"""

import os
import asyncio
import aiohttp
from typing import Dict, Optional, List
from abc import ABC, abstractmethod


class LLMInterface(ABC):
    """Abstract interface for LLM providers"""
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response from the LLM"""
        pass
    
    @abstractmethod
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate a response using chat format"""
        pass


class OpenAIAdapter(LLMInterface):
    """Adapter for OpenAI API (GPT models)"""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = "gpt-4",
                 base_url: str = "https://api.openai.com/v1"):
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.base_url = base_url
        
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate using completion format"""
        messages = [{"role": "user", "content": prompt}]
        return await self.chat(messages, **kwargs)
    
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate using chat format"""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 500)
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API error: {response.status} - {error_text}")
                
                data = await response.json()
                return data["choices"][0]["message"]["content"]


class AnthropicAdapter(LLMInterface):
    """Adapter for Anthropic API (Claude models)"""
    
    def __init__(self,
                 api_key: Optional[str] = None,
                 model: str = "claude-3-sonnet-20240229",
                 base_url: str = "https://api.anthropic.com"):
        
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self.base_url = base_url
        
        if not self.api_key:
            raise ValueError("Anthropic API key not provided")
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate using messages format"""
        messages = [{"role": "user", "content": prompt}]
        return await self.chat(messages, **kwargs)
    
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate using messages format"""
        
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 1024),
            "temperature": kwargs.get("temperature", 0.7)
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/v1/messages",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API error: {response.status} - {error_text}")
                
                data = await response.json()
                return data["content"][0]["text"]


class HuggingFaceAdapter(LLMInterface):
    """Adapter for HuggingFace Inference API"""
    
    def __init__(self,
                 api_key: Optional[str] = None,
                 model: str = "mistralai/Mistral-7B-Instruct-v0.1",
                 base_url: str = "https://api-inference.huggingface.co/models"):
        
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY")
        self.model = model
        self.base_url = base_url
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate using inference API"""
        
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": kwargs.get("temperature", 0.7),
                "max_new_tokens": kwargs.get("max_tokens", 500),
                "return_full_text": False
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/{self.model}",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API error: {response.status} - {error_text}")
                
                data = await response.json()
                return data[0]["generated_text"]
    
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Convert chat to prompt and generate"""
        # Simple conversion - may need adjustment based on model
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        return await self.generate(prompt, **kwargs)


class LocalModelAdapter(LLMInterface):
    """Adapter for locally hosted models (e.g., via Ollama, vLLM)"""
    
    def __init__(self,
                 base_url: str = "http://localhost:11434",
                 model: str = "llama2"):
        
        self.base_url = base_url
        self.model = model
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate using local API"""
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", 0.7),
                "num_predict": kwargs.get("max_tokens", 500)
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API error: {response.status} - {error_text}")
                
                data = await response.json()
                return data["response"]
    
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate using chat endpoint if available"""
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", 0.7)
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                
                if response.status != 200:
                    # Fallback to generate endpoint
                    prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
                    return await self.generate(prompt, **kwargs)
                
                data = await response.json()
                return data["message"]["content"]


class AzureOpenAIAdapter(LLMInterface):
    """Adapter for Azure OpenAI Service"""
    
    def __init__(self,
                 api_key: Optional[str] = None,
                 endpoint: Optional[str] = None,
                 deployment_name: str = "gpt-4",
                 api_version: str = "2024-02-15-preview"):
        
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.deployment_name = deployment_name
        self.api_version = api_version
        
        if not self.api_key or not self.endpoint:
            raise ValueError("Azure OpenAI credentials not provided")
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate using Azure OpenAI"""
        messages = [{"role": "user", "content": prompt}]
        return await self.chat(messages, **kwargs)
    
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate using chat format"""
        
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        url = f"{self.endpoint}/openai/deployments/{self.deployment_name}/chat/completions?api-version={self.api_version}"
        
        payload = {
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 500)
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API error: {response.status} - {error_text}")
                
                data = await response.json()
                return data["choices"][0]["message"]["content"]


def get_llm_adapter(provider: str, **kwargs) -> LLMInterface:
    """Factory function to get appropriate LLM adapter"""
    
    adapters = {
        "openai": OpenAIAdapter,
        "anthropic": AnthropicAdapter,
        "huggingface": HuggingFaceAdapter,
        "local": LocalModelAdapter,
        "azure": AzureOpenAIAdapter
    }
    
    if provider not in adapters:
        raise ValueError(f"Unknown provider: {provider}. Available: {list(adapters.keys())}")
    
    return adapters[provider](**kwargs)


# Example usage
async def test_adapters():
    """Test different LLM adapters"""
    
    # Example with mock/local model
    print("Testing Local Model Adapter...")
    try:
        local_llm = LocalModelAdapter(model="llama2")
        response = await local_llm.generate("Hello, how are you?")
        print(f"Response: {response[:100]}...")
    except Exception as e:
        print(f"Local model not available: {e}")
    
    print("\nAdapter factory test...")
    # You would use actual API keys in production
    # llm = get_llm_adapter("openai", model="gpt-4")
    # response = await llm.generate("Test prompt")


if __name__ == "__main__":
    asyncio.run(test_adapters())
