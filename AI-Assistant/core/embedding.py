import os
import time
import hashlib
import json
import yaml
import numpy as np
import requests
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import threading
from pathlib import Path
from sentence_transformers import SentenceTransformer
from core.logger import get_logger, log_performance, timer
from core.cache import CacheManager, EmbeddingCache

@dataclass
class EmbeddingStats:
    """Statistics for embedding generation and performance."""
    total_requests: int = 0
    total_texts_embedded: int = 0
    total_processing_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    batch_requests: int = 0
    average_batch_size: float = 0.0
    error_count: int = 0
    
    def add_request(self, text_count: int, processing_time: float, 
                   cached: bool = False, batch_size: int = 1):
        """Add a request to the embedding statistics."""
        self.total_requests += 1
        
        if cached:
            self.cache_hits += text_count
        else:
            self.cache_misses += text_count
            self.total_texts_embedded += text_count
            self.total_processing_time += processing_time
        
        if batch_size > 1:
            self.batch_requests += 1
            # Update average batch size
            self.average_batch_size = (
                (self.average_batch_size * (self.batch_requests - 1) + batch_size) 
                / self.batch_requests
            )
    
    @property
    def average_processing_time(self) -> float:
        """Calculate average processing time per request."""
        return self.total_processing_time / max(self.total_requests - (self.cache_hits // max(self.average_batch_size, 1)), 1)
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

class OllamaEmbeddings:
    """
    Ollama embeddings interface for local embedding generation.
    """
    
    def __init__(self, model_name: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        """Initialize Ollama embeddings interface."""
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.api_url = f"{self.base_url}/api"
        
        # Test connection and model availability
        self._test_connection()
    
    def _test_connection(self):
        """Test connection to Ollama server and check model availability."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                available_models = [model['name'] for model in models]
                
                # Check if our embedding model is available
                model_available = any(self.model_name in model for model in available_models)
                if not model_available:
                    print(f"Warning: Embedding model '{self.model_name}' not found in Ollama.")
                    print(f"Available models: {available_models}")
                    print(f"To install the embedding model, run: ollama pull {self.model_name}")
            else:
                raise Exception(f"Ollama server responded with status {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"Cannot connect to Ollama server at {self.base_url}. Is Ollama running? Error: {e}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents using Ollama."""
        embeddings = []
        
        for text in texts:
            embedding = self.embed_query(text)
            embeddings.append(embedding)
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single text using Ollama API."""
        try:
            payload = {
                "model": self.model_name,
                "prompt": text
            }
            
            response = requests.post(
                f"{self.api_url}/embeddings",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                embedding = result.get('embedding', [])
                if not embedding:
                    raise Exception("No embedding returned from Ollama")
                return embedding
            else:
                raise Exception(f"Ollama embeddings API error: {response.status_code} - {response.text}")
                
        except requests.exceptions.Timeout:
            raise Exception("Ollama embeddings request timed out")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request to Ollama embeddings failed: {e}")

class EnhancedEmbeddingModel:
    """
    Enhanced embedding model with support for both Ollama and SentenceTransformers.
    """
    
    def __init__(self, config_path: str = "config.yaml", 
                 cache_manager: Optional[CacheManager] = None):
        """Initialize the enhanced embedding model."""
        self.config_path = config_path
        self.config = self._load_embedding_config(config_path)
        self.logger = get_logger()
        
        # Initialize cache
        self.cache_manager = cache_manager
        self.embedding_cache = EmbeddingCache(cache_manager) if cache_manager else None
        
        # Statistics and monitoring
        self.stats = EmbeddingStats()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize the embedding model based on provider
        self.provider = self.config.get("embedding", {}).get("provider", "sentence_transformers")
        self.model = self._create_embedding_model()
        self.model_info = self._get_model_info()
        
        # Batch processing settings
        self.batch_size = self.config.get("batch_processing", {}).get("batch_size", 100)
        self.max_batch_size = self.config.get("batch_processing", {}).get("max_batch_size", 500)
        
        self.logger.log("Enhanced embedding model initialized", 
                       component="embedding",
                       extra={
                           "provider": self.provider,
                           "model": self.model_info.get("model_name", "unknown"),
                           "cache_enabled": self.embedding_cache is not None
                       })
    
    def _load_embedding_config(self, config_path: str) -> Dict[str, Any]:
        """Load embedding configuration from YAML file."""
        default_config = {
            "embedding": {
                "provider": "sentence_transformers",  # or "ollama"
                "model": "all-MiniLM-L6-v2",
                "ollama_model": "nomic-embed-text",
                "ollama_base_url": "http://localhost:11434",
                "dimensions": 384,
                "chunk_size": 1000,
                "max_retries": 3
            },
            "batch_processing": {
                "batch_size": 100,
                "max_batch_size": 500,
                "parallel_processing": False
            },
            "caching": {
                "enable_embedding_cache": True,
                "cache_ttl": 86400,  # 24 hours
            },
            "optimization": {
                "enable_batch_optimization": True,
                "deduplicate_inputs": True,
                "normalize_whitespace": True
            }
        }
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    file_config = yaml.safe_load(f) or {}
                    
                    # Extract embedding config from different sections
                    if "embedding" in file_config:
                        default_config["embedding"].update(file_config["embedding"])
                    
                    # Check for Ollama-specific config
                    if "ollama" in file_config and "embedding" in file_config["ollama"]:
                        default_config["embedding"].update(file_config["ollama"]["embedding"])
                    
                    # Performance and cache configs
                    if "performance" in file_config:
                        perf_config = file_config["performance"]
                        if "batch" in perf_config:
                            default_config["batch_processing"].update(perf_config["batch"])
                    
                    if "cache" in file_config:
                        cache_config = file_config["cache"]
                        default_config["caching"].update(cache_config)
                        
        except Exception as e:
            self.logger.log(f"Error loading embedding config: {e}", 
                           level="WARNING", component="embedding")
        
        return default_config
    
    def _create_embedding_model(self):
        """Create embedding model based on provider configuration."""
        provider = self.config.get("embedding", {}).get("provider", "sentence_transformers")
        
        if provider == "ollama":
            try:
                model_name = self.config["embedding"].get("ollama_model", "nomic-embed-text")
                base_url = self.config["embedding"].get("ollama_base_url", "http://localhost:11434")
                
                model = OllamaEmbeddings(model_name=model_name, base_url=base_url)
                self.logger.log("Ollama embedding model initialized", component="embedding")
                return model
                
            except Exception as e:
                self.logger.log(f"Ollama embedding model failed, falling back to SentenceTransformers: {e}", 
                               level="WARNING", component="embedding")
                # Fall back to SentenceTransformers
                return self._create_sentence_transformer()
        
        else:
            # Use SentenceTransformers (default)
            return self._create_sentence_transformer()
    
    def _create_sentence_transformer(self):
        """Create SentenceTransformer model as fallback."""
        model_name = self.config.get("embedding", {}).get("model", "all-MiniLM-L6-v2")
        
        try:
            model = SentenceTransformer(model_name)
            self.logger.log("SentenceTransformer embedding model initialized", component="embedding")
            return model
        except Exception as e:
            self.logger.log(f"Error creating SentenceTransformer model: {e}", 
                           level="ERROR", component="embedding")
            # Return a mock model that doesn't crash
            return MockEmbeddingModel()
        
    def _get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        if self.provider == "ollama":
            return {
                "provider": "ollama",
                "model_name": self.config["embedding"]["ollama_model"],
                "dimensions": self.config["embedding"].get("dimensions", 768),  # Ollama models typically 768
                "cost_per_1k_tokens": 0.0  # Local
            }
        else:
            return {
                "provider": "sentence_transformers",
                "model_name": self.config["embedding"]["model"],
                "dimensions": self.config["embedding"].get("dimensions", 384),
                "cost_per_1k_tokens": 0.0  # Free
            }
    
    def embed_documents(self, texts: List[str], 
                       use_cache: bool = True,
                       batch_size: Optional[int] = None) -> List[List[float]]:
        """
        Embed documents with intelligent caching and batch processing.
        """
        if not texts:
            return []
        
        start_time = time.time()
        actual_batch_size = batch_size or self.batch_size
        
        # Preprocess texts
        processed_texts = self._preprocess_texts(texts)
        
        # Handle caching and deduplication
        embeddings_result = []
        texts_to_embed = []
        cache_map = {}  # Map from original index to cache result
        embed_map = {}  # Map from original index to embedding index
        
        for i, text in enumerate(processed_texts):
            if use_cache and self.embedding_cache:
                cached_embedding = self.embedding_cache.get_embedding(text, self.model_info["model_name"])
                if cached_embedding is not None:
                    embeddings_result.append(cached_embedding)
                    cache_map[i] = len(embeddings_result) - 1
                    continue
            
            # Check for duplicates in current batch
            if self.config.get("optimization", {}).get("deduplicate_inputs", True):
                try:
                    existing_idx = texts_to_embed.index(text)
                    embed_map[i] = existing_idx
                    embeddings_result.append(None)  # Placeholder
                    continue
                except ValueError:
                    pass
            
            embed_map[i] = len(texts_to_embed)
            texts_to_embed.append(text)
            embeddings_result.append(None)  # Placeholder
        
        # Process texts that need embedding
        new_embeddings = []
        if texts_to_embed:
            new_embeddings = self._embed_batch(texts_to_embed, actual_batch_size)
            
            # Store in cache
            if use_cache and self.embedding_cache:
                for text, embedding in zip(texts_to_embed, new_embeddings):
                    self.embedding_cache.set_embedding(
                        text, 
                        self.model_info["model_name"], 
                        embedding,
                        ttl=self.config.get("caching", {}).get("cache_ttl", 86400)
                    )
        
        # Reconstruct final embeddings list
        final_embeddings = []
        for i, placeholder in enumerate(embeddings_result):
            if i in cache_map:
                # Use cached result
                final_embeddings.append(embeddings_result[cache_map[i]])
            elif i in embed_map:
                # Use newly computed embedding
                embed_idx = embed_map[i]
                if embed_idx < len(new_embeddings):
                    final_embeddings.append(new_embeddings[embed_idx])
                else:
                    # Handle deduplication
                    original_text = processed_texts[i]
                    for j, text in enumerate(texts_to_embed):
                        if text == original_text:
                            final_embeddings.append(new_embeddings[j])
                            break
        
        processing_time = time.time() - start_time
        
        # Update statistics
        cache_hits = len([i for i in cache_map])
        
        with self._lock:
            self.stats.add_request(
                text_count=len(texts),
                processing_time=processing_time,
                cached=len(texts_to_embed) == 0,
                batch_size=len(texts_to_embed) if texts_to_embed else 1
            )
        
        # Log performance
        self.logger.log_performance(
            "embedding_documents",
            processing_time,
            "embedding",
            {
                "text_count": len(texts),
                "texts_embedded": len(texts_to_embed),
                "cache_hits": cache_hits,
                "cache_misses": len(texts_to_embed),
                "batch_size": actual_batch_size,
                "provider": self.provider
            }
        )
        
        return final_embeddings
    
    def embed_query(self, text: str, use_cache: bool = True) -> List[float]:
        """
        Embed a single query with caching.
        """
        start_time = time.time()
        
        # Preprocess text
        processed_text = self._preprocess_text(text)
        
        # Check cache first
        if use_cache and self.embedding_cache:
            cached_embedding = self.embedding_cache.get_embedding(
                processed_text, 
                self.model_info["model_name"]
            )
            if cached_embedding is not None:
                processing_time = time.time() - start_time
                
                with self._lock:
                    self.stats.add_request(1, processing_time, cached=True)
                
                self.logger.log("Query embedding cache hit", 
                               component="embedding",
                               extra={"text_preview": text[:50], "provider": self.provider})
                return cached_embedding
        
        # Generate embedding
        try:
            with timer(f"query_embedding_{self.provider}", "embedding"):
                if self.provider == "ollama":
                    embedding = self.model.embed_query(processed_text)
                else:
                    embedding = self.model.encode(processed_text).tolist()
            
            # Cache the result
            if use_cache and self.embedding_cache:
                self.embedding_cache.set_embedding(
                    processed_text,
                    self.model_info["model_name"],
                    embedding,
                    ttl=self.config.get("caching", {}).get("cache_ttl", 86400)
                )
            
            processing_time = time.time() - start_time
            
            # Update statistics
            with self._lock:
                self.stats.add_request(1, processing_time, cached=False)
            
            self.logger.log("Query embedding generated", 
                           component="embedding",
                           extra={
                               "text_preview": text[:50],
                               "processing_time": processing_time,
                               "provider": self.provider
                           })
            
            return embedding
            
        except Exception as e:
            with self._lock:
                self.stats.error_count += 1
            
            self.logger.log(f"Error generating query embedding: {e}", 
                           level="ERROR", component="embedding")
            raise
    
    def __call__(self, text: str) -> List[float]:
        """Make the embedding model callable for FAISS compatibility."""
        return self.embed_query(text)

    def _embed_batch(self, texts: List[str], batch_size: int) -> List[List[float]]:
        """Embed texts in batches."""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Generate embeddings for this batch
            try:
                with timer(f"batch_embedding_{self.provider}_{len(batch_texts)}", "embedding"):
                    if self.provider == "ollama":
                        # Ollama processes one at a time currently
                        batch_embeddings = []
                        for text in batch_texts:
                            embedding = self.model.embed_query(text)
                            batch_embeddings.append(embedding)
                    else:
                        # SentenceTransformers can do true batch processing
                        batch_embeddings = self.model.encode(batch_texts).tolist()
                
                all_embeddings.extend(batch_embeddings)
                
                self.logger.log(f"Batch embedding completed", 
                               component="embedding",
                               extra={
                                   "batch_size": len(batch_texts),
                                   "batch_number": i//batch_size + 1,
                                   "provider": self.provider
                               })
                
            except Exception as e:
                with self._lock:
                    self.stats.error_count += 1
                
                self.logger.log(f"Error in batch embedding: {e}", 
                               level="ERROR", component="embedding")
                raise
        
        return all_embeddings
    
    def _preprocess_texts(self, texts: List[str]) -> List[str]:
        """Preprocess texts for embedding generation."""
        if not self.config.get("optimization", {}).get("normalize_whitespace", True):
            return texts
        
        processed = []
        for text in texts:
            processed_text = self._preprocess_text(text)
            processed.append(processed_text)
        
        return processed
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess a single text for embedding generation."""
        if not text:
            return ""
        
        # Normalize whitespace
        if self.config.get("optimization", {}).get("normalize_whitespace", True):
            text = " ".join(text.split())
        
        return text.strip()
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get comprehensive embedding statistics."""
        with self._lock:
            return {
                "requests": {
                    "total": self.stats.total_requests,
                    "errors": self.stats.error_count,
                    "success_rate": (self.stats.total_requests - self.stats.error_count) / max(self.stats.total_requests, 1)
                },
                "performance": {
                    "total_texts_embedded": self.stats.total_texts_embedded,
                    "average_processing_time": self.stats.average_processing_time,
                    "cache_hit_rate": self.stats.cache_hit_rate
                },
                "caching": {
                    "cache_hits": self.stats.cache_hits,
                    "cache_misses": self.stats.cache_misses,
                    "cache_enabled": self.embedding_cache is not None
                },
                "batching": {
                    "batch_requests": self.stats.batch_requests,
                    "average_batch_size": self.stats.average_batch_size
                },
                "model_info": self.model_info
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current embedding model."""
        return {
            **self.model_info,
            "config": {
                "batch_size": self.batch_size,
                "max_batch_size": self.max_batch_size,
                "cache_enabled": self.embedding_cache is not None,
                "provider": self.provider
            }
        }
    
    def reset_stats(self):
        """Reset embedding statistics."""
        with self._lock:
            self.stats = EmbeddingStats()
            self.logger.log("Embedding statistics reset", component="embedding")
    
    def validate_embeddings(self, embeddings: List[List[float]]) -> Dict[str, Any]:
        """Validate embedding quality and consistency."""
        if not embeddings:
            return {"valid": False, "error": "No embeddings provided"}
        
        # Check dimensions consistency
        dimensions = set(len(emb) for emb in embeddings)
        if len(dimensions) > 1:
            return {
                "valid": False,
                "error": f"Inconsistent embedding dimensions: {dimensions}"
            }
        
        expected_dim = self.model_info.get("dimensions", 384)
        actual_dim = next(iter(dimensions))
        
        # Check for zero vectors (potentially problematic)
        zero_vectors = sum(1 for emb in embeddings if all(x == 0 for x in emb))
        
        # Check for NaN or infinite values
        invalid_vectors = 0
        for emb in embeddings:
            if any(np.isnan(x) or np.isinf(x) for x in emb):
                invalid_vectors += 1
        
        return {
            "valid": invalid_vectors == 0,
            "total_embeddings": len(embeddings),
            "embedding_dimension": actual_dim,
            "expected_dimension": expected_dim,
            "zero_vectors": zero_vectors,
            "invalid_vectors": invalid_vectors,
            "quality_score": 1.0 - (zero_vectors + invalid_vectors) / len(embeddings),
            "provider": self.provider
        }

class MockEmbeddingModel:
    """Mock embedding model for fallback when both providers fail."""
    
    def encode(self, texts):
        """Return random embeddings for testing."""
        if isinstance(texts, str):
            return np.random.rand(384).tolist()
        return [np.random.rand(384).tolist() for _ in texts]
    
    def embed_query(self, text: str):
        """Return random embedding for single query."""
        return np.random.rand(384).tolist()
    
    def embed_documents(self, texts: List[str]):
        """Return random embeddings for documents."""
        return [np.random.rand(384).tolist() for _ in texts]

# Global embedding model instance
_enhanced_embedding_model = None

def get_embedding_model(config_path: str = "config.yaml", 
                       cache_manager: Optional[CacheManager] = None) -> EnhancedEmbeddingModel:
    """Get the global enhanced embedding model instance."""
    global _enhanced_embedding_model
    if _enhanced_embedding_model is None:
        _enhanced_embedding_model = EnhancedEmbeddingModel(config_path, cache_manager)
    return _enhanced_embedding_model

def get_simple_embedding_model():
    """Get the underlying embedding model for backward compatibility."""
    enhanced_model = get_embedding_model()
    return enhanced_model.model

# Convenience functions
def embed_documents(texts: List[str], use_cache: bool = True) -> List[List[float]]:
    """Embed documents using the enhanced model."""
    model = get_embedding_model()
    return model.embed_documents(texts, use_cache=use_cache)

def embed_query(text: str, use_cache: bool = True) -> List[float]:
    """Embed a query using the enhanced model."""
    model = get_embedding_model()
    return model.embed_query(text, use_cache=use_cache)

def get_embedding_stats() -> Dict[str, Any]:
    """Get embedding statistics."""
    model = get_embedding_model()
    return model.get_embedding_stats()

def check_ollama_embeddings_status() -> Dict[str, Any]:
    """Check if Ollama embeddings are available."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            embedding_models = [model['name'] for model in models if 'embed' in model['name'].lower()]
            return {
                "status": "available",
                "embedding_models": embedding_models,
                "total_models": len(models)
            }
        else:
            return {
                "status": "server_error",
                "error": f"Server responded with status {response.status_code}"
            }
    except requests.exceptions.RequestException as e:
        return {
            "status": "unavailable",
            "error": str(e),
            "suggestion": "Start Ollama with: ollama serve, then install embedding model: ollama pull nomic-embed-text"
        }

def estimate_embedding_cost(texts: List[str], model_name: str = "ollama") -> Dict[str, Any]:
    """
    Estimate embedding cost (for Ollama it's free, but keeping for compatibility).
    
    Args:
        texts: List of texts to embed
        model_name: Model name (ignored for Ollama)
    
    Returns:
        Cost estimation (always $0 for local models)
    """
    return {
        "provider": "ollama",
        "total_texts": len(texts),
        "estimated_cost_usd": 0.0,
        "cost_per_1k_tokens": 0.0,
        "note": "Local Ollama models are free"
    }