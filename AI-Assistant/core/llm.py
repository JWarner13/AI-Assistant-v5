import os
import time
import json
import yaml
import hashlib
import traceback
import requests
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import threading
from dotenv import load_dotenv
load_dotenv()
from core.logger import get_logger, log_performance, timer
from core.cache import CacheManager


@dataclass
class LLMUsageStats:
    """Track LLM usage statistics for performance monitoring."""
    total_requests: int = 0
    average_response_time: float = 0.0
    error_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    
    def add_request(self, response_time: float, cached: bool = False):
        """Add a request to the usage statistics."""
        self.total_requests += 1
        if not cached:
            self.cache_misses += 1
        else:
            self.cache_hits += 1
        
        # Update average response time
        if self.total_requests > 0:
            self.average_response_time = (
                (self.average_response_time * (self.total_requests - 1) + response_time) 
                / self.total_requests
            )

class OllamaLLM:
    """
    Ollama LLM interface for local model execution.
    """
    
    def __init__(self, model_name: str = "llama3.2", base_url: str = "http://localhost:11434"):
        """Initialize Ollama LLM interface."""
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.api_url = f"{self.base_url}/api"
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test connection to Ollama server."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                available_models = [model['name'] for model in models]
                
                # Check if our model is available
                model_available = any(self.model_name in model for model in available_models)
                if not model_available:
                    print(f"Warning: Model '{self.model_name}' not found in Ollama. Available models: {available_models}")
                    print(f"To install the model, run: ollama pull {self.model_name}")
            else:
                raise Exception(f"Ollama server responded with status {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"Cannot connect to Ollama server at {self.base_url}. Is Ollama running? Error: {e}")
    
    def invoke(self, prompt: str, **kwargs) -> str:
        """Generate response using Ollama API."""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", 0.1),
                    "top_p": kwargs.get("top_p", 0.9),
                    "num_predict": kwargs.get("max_tokens", 512)
                }
            }
            
            response = requests.post(
                f"{self.api_url}/generate",
                json=payload,
                timeout=120  # 2 minute timeout for local models
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
                
        except requests.exceptions.Timeout:
            raise Exception("Ollama request timed out. The model might be too large or the prompt too complex.")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request to Ollama failed: {e}")

class EnhancedLLM:
    """
    Enhanced LLM interface using Ollama models with caching and monitoring.
    """
    
    def __init__(self, config_path: str = "config.yaml", 
                 cache_manager: Optional[CacheManager] = None):
        """Initialize the enhanced LLM interface."""
        self.config_path = config_path
        self.config = self._load_llm_config(config_path)
        self.logger = get_logger()
        self.cache_manager = cache_manager
        self.usage_stats = LLMUsageStats()
        
        # Initialize model
        self.primary_model = self._initialize_model()
        
        # Response templates for different query types
        self.response_templates = self._load_response_templates()
        
        self.logger.log("Enhanced LLM interface initialized with Ollama", 
                       component="llm", 
                       extra={"model": self.config["llm"]["model"]})
    
    def _load_llm_config(self, config_path: str) -> Dict[str, Any]:
        """Load LLM configuration from YAML file."""
        default_config = {
            "llm": {
                "provider": "ollama",
                "model": "llama3.2",  # Default Ollama model
                "base_url": "http://localhost:11434",
                "temperature": 0.1,
                "max_tokens": 512,
                "max_retries": 3
            },
            "caching": {
                "enable_response_caching": True,
                "cache_ttl": 3600
            }
        }
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    file_config = yaml.safe_load(f) or {}
                    
                    # Check for Ollama-specific config
                    ollama_config = file_config.get("ollama", {})
                    if "llm" in ollama_config:
                        default_config["llm"].update(ollama_config["llm"])
                    
                    # Also check for top-level cache config
                    if "cache" in file_config:
                        default_config["caching"].update(file_config["cache"])
                        
        except Exception as e:
            self.logger.log(f"Error loading LLM config: {e}", level="WARNING", component="llm")
        
        return default_config
    
    def _initialize_model(self):
        """Initialize the Ollama model with better error handling."""
        try:
            llm_config = self.config["llm"]
            
            # Initialize Ollama model
            model = OllamaLLM(
                model_name=llm_config.get("model", "llama3.2"),
                base_url=llm_config.get("base_url", "http://localhost:11434")
            )
            
            # Test the model with a simple call
            test_response = model.invoke("Test")
            self.logger.log("Ollama model initialized successfully", 
                           component="llm",
                           extra={"model": llm_config.get("model"), "test_response_length": len(test_response)})
            return model
                
        except Exception as e:
            self.logger.log(f"Ollama model initialization failed: {e}", level="ERROR", component="llm")
            # Return a mock model that at least doesn't crash
            return MockLLM()
    
    def _load_response_templates(self) -> Dict[str, str]:
        """Load response templates for different query types."""
        return {
            "default": """Based on the following context, provide a clear and accurate answer to the question.

Context: {context}

Question: {query}

Answer:"""
        }
    
    def invoke(self, prompt: str, 
                query_type: str = "default",
                use_cache: bool = True,
                max_retries: Optional[int] = None) -> str:
        """
        Generate a response using the LLM with enhanced features.
        """
        start_time = time.time()
        
        # Check cache first
        if use_cache and self.cache_manager:
            cache_key = self._generate_cache_key(prompt, query_type)
            cached_response = self.cache_manager.get(cache_key)
            if cached_response:
                self.usage_stats.add_request(time.time() - start_time, cached=True)
                self.logger.log("LLM cache hit", component="llm")
                return cached_response
        
        # Generate response with simplified retry logic
        response = self._generate_response_with_retry(
            prompt, max_retries or self.config["llm"]["max_retries"]
        )
        
        processing_time = time.time() - start_time
        
        # Update statistics
        self.usage_stats.add_request(processing_time)
        
        # Cache the response
        if use_cache and self.cache_manager:
            cache_ttl = self.config.get("caching", {}).get("cache_ttl", 3600)
            self.cache_manager.set(cache_key, response, cache_ttl)
        
        return response
    
    def _generate_response_with_retry(self, prompt: str, max_retries: int) -> str:
        """Generate response with simplified retry logic."""
        last_error = None
        
        # Truncate prompt if too long (local model limitation)
        if len(prompt) > 4000:  # Conservative limit for local models
            prompt = prompt[:4000] + "..."
            self.logger.log("Prompt truncated for local model", component="llm", 
                           extra={"original_length": len(prompt), "truncated_length": 4000})
        
        for attempt in range(max_retries):
            try:
                self.logger.log(
                    f"Attempting Ollama invoke (attempt {attempt + 1})",
                    component="llm",
                    extra={"attempt": attempt + 1, "prompt_length": len(prompt)}
                )
                
                response = self.primary_model.invoke(
                    prompt,
                    temperature=self.config["llm"].get("temperature", 0.1),
                    max_tokens=self.config["llm"].get("max_tokens", 512)
                )
                
                if response and len(response.strip()) > 0:
                    self.logger.log(f"Ollama response generated successfully", 
                                   component="llm", 
                                   extra={"attempt": attempt + 1, "response_length": len(response)})
                    return response.strip()
                else:
                    raise Exception("Empty response from Ollama model")
                
            except Exception as e:
                last_error = e
                self.usage_stats.error_count += 1

                # Full traceback for debugging
                error_trace = traceback.format_exc()
                self.logger.log(
                    f"Ollama call failed (attempt {attempt + 1}): {e}\n{error_trace}",
                    level="WARNING",
                    component="llm"
                )
                
                if attempt < max_retries - 1:
                    time.sleep(2)  # Longer wait for local models
        
        # If all retries failed, return a fallback response
        fallback_response = f"I apologize, but I'm having trouble processing your request right now. The local Ollama model is experiencing issues: {str(last_error)}. Please ensure Ollama is running and the model '{self.config['llm']['model']}' is installed."
        
        self.logger.log(f"Ollama failed after {max_retries} retries, returning fallback", 
                       level="ERROR", component="llm")
        return fallback_response
    
    def _generate_cache_key(self, prompt: str, query_type: str) -> str:
        """Generate a cache key for the prompt."""
        content = f"{prompt}|{query_type}|{self.config['llm']['model']}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def generate_with_template(self, context: str, query: str, query_type: str = "default") -> str:
        """Generate response using appropriate template for query type."""
        template = self.response_templates.get(query_type, self.response_templates["default"])
        prompt = template.format(context=context, query=query)
        
        return self.invoke(prompt, query_type=query_type)
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get comprehensive usage statistics."""
        cache_total = self.usage_stats.cache_hits + self.usage_stats.cache_misses
        cache_hit_rate = self.usage_stats.cache_hits / cache_total if cache_total > 0 else 0
        
        return {
            "requests": {
                "total": self.usage_stats.total_requests,
                "errors": self.usage_stats.error_count,
                "success_rate": (self.usage_stats.total_requests - self.usage_stats.error_count) / max(self.usage_stats.total_requests, 1)
            },
            "performance": {
                "average_response_time": self.usage_stats.average_response_time,
                "cache_hit_rate": cache_hit_rate,
                "cache_hits": self.usage_stats.cache_hits,
                "cache_misses": self.usage_stats.cache_misses
            },
            "model_info": {
                "provider": "ollama",
                "model": self.config["llm"]["model"],
                "base_url": self.config["llm"]["base_url"]
            }
        }

class MockLLM:
    """Mock LLM for fallback when Ollama fails."""
    
    def invoke(self, prompt: str, **kwargs) -> str:
        return "I'm a fallback response. The Ollama model is currently unavailable. Please ensure Ollama is running and the required model is installed. Run 'ollama list' to see available models or 'ollama pull llama3.2' to install a model."

# Global LLM instance
_enhanced_llm = None

def get_llm(config_path: str = "config.yaml", cache_manager: Optional[CacheManager] = None) -> EnhancedLLM:
    """Get the global enhanced LLM instance."""
    global _enhanced_llm
    if _enhanced_llm is None:
        _enhanced_llm = EnhancedLLM(config_path, cache_manager)
    return _enhanced_llm

def get_simple_llm():
    """Get a simple LLM instance for backward compatibility."""
    enhanced_llm = get_llm()
    return enhanced_llm.primary_model

# Convenience functions
def generate_response(context: str, query: str, query_type: str = "default") -> str:
    """Generate a response using the enhanced LLM with templates."""
    llm = get_llm()
    return llm.generate_with_template(context, query, query_type)

def get_llm_stats() -> Dict[str, Any]:
    """Get LLM usage statistics."""
    llm = get_llm()
    return llm.get_usage_stats()

def check_ollama_status() -> Dict[str, Any]:
    """Check Ollama server status and available models."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            return {
                "status": "running",
                "models": [model['name'] for model in models],
                "model_count": len(models)
            }
        else:
            return {
                "status": "error",
                "error": f"Server responded with status {response.status_code}"
            }
    except requests.exceptions.RequestException as e:
        return {
            "status": "not_running",
            "error": str(e),
            "suggestion": "Start Ollama with: ollama serve"
        }

def estimate_query_cost(query: str, model_name: str = "ollama") -> Dict[str, Any]:
    """
    Estimate query cost (for Ollama it's free, but keeping for compatibility).
    
    Args:
        query: Query string
        model_name: Model name (ignored for Ollama)
    
    Returns:
        Cost estimation (always $0 for local models)
    """
    return {
        "provider": "ollama",
        "query_length": len(query),
        "estimated_cost_usd": 0.0,
        "cost_per_1k_tokens": 0.0,
        "note": "Local Ollama models are free"
    }