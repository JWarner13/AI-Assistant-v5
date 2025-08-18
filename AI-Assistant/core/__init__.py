import sys
import os
from typing import Dict, Any, Optional, List
from pathlib import Path

# Package metadata - Local AI Focus
__version__ = "2.0.0-local"
__author__ = "AI Document Assistant Team"
__description__ = "Privacy-first RAG assistant with local Ollama models - no cloud dependencies"
__license__ = "MIT"

# Minimum Python version check
MIN_PYTHON = (3, 8)
if sys.version_info < MIN_PYTHON:
    sys.exit(f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or later is required.")

# Package-level configuration for local execution
_package_config = {
    "initialized": False,
    "config_path": "config.yaml",
    "cache_manager": None,
    "logger": None,
    "performance_tracking": True,
    "debug_mode": False,
    "local_mode": True,
    "privacy_mode": True,
    "offline_ready": True
}

# Local execution banner
def display_local_banner():
    """Display local execution information."""
    print("🏠 AI Document Assistant - Local Edition")
    print("🔒 100% Private • 💰 $0 API Costs • 🌐 Offline Ready")
    print("=" * 50)

# Core module imports with error handling
try:
    from .cache import CacheManager
    from .rag_engine import run_assistant, run_batch_queries, _generate_cache_key
    
    # Import other modules
    from .document_loader import (
        load_documents,
        extract_text_from_pdf,
        extract_text_from_txt,
        split_into_chunks,
        get_document_summary,
        validate_documents
    )
    
    from .embedding import (
        get_embedding_model,
        embed_documents,
        embed_query,
        get_embedding_stats,
        estimate_embedding_cost,
        EnhancedEmbeddingModel
    )
    
    from .index import (
        get_index_manager,
        build_index,
        search_index,
        build_and_search,
        get_index_stats,
        IndexManager
    )
    
    from .llm import (
        get_llm,
        generate_response,
        get_llm_stats,
        estimate_query_cost,
        EnhancedLLM
    )
    
    from .reasoning import (
        explain_reasoning,
        detect_conflicts,
        perform_multi_hop_reasoning
    )
    
    from .logger import (
        get_logger,
        log,
        log_performance,
        log_error,
        log_debug,
        get_performance_summary,
        timer,
        log_execution_time,
        log_method_calls,
        EnhancedLogger
    )
    
    from .cache import (
        CacheManager,
        EmbeddingCache,
        QueryCache
    )
    
    from .output_formatter import (
        ResponseFormatter,
        BatchFormatter,
        format_output,
        format_for_api,
        format_for_human,
        format_for_report,
        format_for_business
    )

except ImportError as e:
    print(f"Warning: Could not import all core modules: {e}")
    print("Some functionality may not be available.")

def get_local_system_info() -> Dict[str, Any]:
    """Get local system information for optimization."""
    try:
        # Try to import psutil for system info
        import psutil
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('.')
        cpu_count = psutil.cpu_count()
        
        return {
            "memory": {
                "total_gb": round(memory.total / (1024**3), 1),
                "available_gb": round(memory.available / (1024**3), 1),
                "percent_used": memory.percent
            },
            "cpu": {
                "cores": cpu_count,
                "logical_cores": psutil.cpu_count(logical=True)
            },
            "disk": {
                "free_gb": round(disk.free / (1024**3), 1),
                "total_gb": round(disk.total / (1024**3), 1)
            }
        }
    except ImportError:
        # Fallback if psutil not available
        return {
            "memory": {"total_gb": 8, "available_gb": 4, "percent_used": 50},
            "cpu": {"cores": 4, "logical_cores": 8},
            "disk": {"free_gb": 100, "total_gb": 500},
            "note": "System info unavailable - install psutil for detailed metrics"
        }
    except Exception:
        return {"error": "Could not detect system information"}

def recommend_ollama_models(system_info: Dict[str, Any]) -> Dict[str, List[str]]:
    """Recommend Ollama models based on local system capabilities."""
    memory_gb = system_info.get("memory", {}).get("total_gb", 8)
    
    recommendations = {
        "recommended": [],
        "alternative": [],
        "advanced": []
    }
    
    if memory_gb >= 32:
        recommendations["recommended"] = ["llama3.1:8b", "nomic-embed-text"]
        recommendations["alternative"] = ["mistral:7b", "deepseek-coder:6.7b"]
        recommendations["advanced"] = ["llama3.1:70b"]
    elif memory_gb >= 16:
        recommendations["recommended"] = ["llama3.2", "nomic-embed-text"]
        recommendations["alternative"] = ["mistral", "llama3.1"]
        recommendations["advanced"] = ["llama3.1:8b"]
    elif memory_gb >= 8:
        recommendations["recommended"] = ["llama3.2:1b", "nomic-embed-text"]
        recommendations["alternative"] = ["llama3.2:3b", "mistral:3b"]
        recommendations["advanced"] = ["llama3.2"]
    else:
        recommendations["recommended"] = ["llama3.2:1b", "nomic-embed-text"]
        recommendations["alternative"] = []
        recommendations["advanced"] = []
    
    return recommendations

def check_ollama_local_setup() -> Dict[str, Any]:
    """Comprehensive local Ollama setup check with system optimization."""
    status = {
        "ollama_running": False,
        "models_available": [],
        "embedding_models": [],
        "system_info": {},
        "recommendations": {},
        "setup_complete": False,
        "issues": [],
        "suggestions": [],
        "local_ready": False
    }
    
    # Get system information
    status["system_info"] = get_local_system_info()
    status["recommendations"] = recommend_ollama_models(status["system_info"])
    
    try:
        import requests
        # Check Ollama server
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            status["ollama_running"] = True
            models_data = response.json().get('models', [])
            status["models_available"] = [model['name'] for model in models_data]
            
            # Categorize models
            llm_models = []
            embedding_models = []
            
            for model_name in status["models_available"]:
                if any(x in model_name.lower() for x in ['llama', 'mistral', 'deepseek', 'codellama']):
                    llm_models.append(model_name)
                elif 'embed' in model_name.lower():
                    embedding_models.append(model_name)
            
            status["llm_models"] = llm_models
            status["embedding_models"] = embedding_models
            
            # Check completeness
            has_llm = len(llm_models) > 0
            has_embed = len(embedding_models) > 0
            
            if not has_llm:
                status["issues"].append("No LLM model found")
                recommended_llm = status["recommendations"]["recommended"][0]
                status["suggestions"].append(f"Install LLM: ollama pull {recommended_llm}")
            
            if not has_embed:
                status["issues"].append("No embedding model found")
                status["suggestions"].append("Install embeddings: ollama pull nomic-embed-text")
            
            status["setup_complete"] = has_llm and has_embed
            status["local_ready"] = status["setup_complete"]
            
        else:
            status["issues"].append(f"Ollama server error: {response.status_code}")
            status["suggestions"].append("Restart Ollama: ollama serve")
            
    except Exception as e:
        status["issues"].append("Cannot connect to local Ollama server")
        status["suggestions"].extend([
            "1. Install Ollama: curl -fsSL https://ollama.com/install.sh | sh",
            "2. Start server: ollama serve",
            "3. Install models: ollama pull llama3.2 && ollama pull nomic-embed-text"
        ])
    
    return status

def initialize_package(config_path: str = "config.yaml", 
                      enable_caching: bool = True,
                      enable_logging: bool = True,
                      debug_mode: bool = False,
                      show_banner: bool = True) -> Dict[str, Any]:
    """Initialize the AI Document Assistant package for local Ollama execution."""
    global _package_config
    
    if show_banner:
        display_local_banner()
    
    init_status = {
        "success": True,
        "components": {},
        "errors": [],
        "warnings": [],
        "local_info": {},
        "privacy_mode": True,
        "offline_ready": True
    }
    
    try:
        # Update package config for local execution
        _package_config.update({
            "config_path": config_path,
            "debug_mode": debug_mode,
            "local_mode": True,
            "privacy_mode": True,
            "offline_ready": True
        })
        
        # Get local system information
        init_status["local_info"] = get_local_system_info()
        
        # Initialize logging system
        if enable_logging:
            try:
                logger = get_logger(config_path)
                _package_config["logger"] = logger
                init_status["components"]["logger"] = "initialized"
            except Exception as e:
                init_status["errors"].append(f"Logger initialization failed: {e}")
                init_status["components"]["logger"] = "failed"
        
        # Initialize local cache manager
        if enable_caching:
            try:
                cache_manager = CacheManager(max_size=2000, default_ttl=7200)
                _package_config["cache_manager"] = cache_manager
                init_status["components"]["cache"] = "initialized"
            except Exception as e:
                init_status["errors"].append(f"Cache initialization failed: {e}")
                init_status["components"]["cache"] = "failed"
        
        # Validate configuration file
        if os.path.exists(config_path):
            init_status["components"]["config"] = "loaded"
        else:
            init_status["warnings"].append(f"Configuration file not found: {config_path}")
            init_status["components"]["config"] = "default"
        
        # Check Ollama setup
        ollama_status = check_ollama_local_setup()
        init_status["ollama_status"] = ollama_status
        
        if not ollama_status["local_ready"]:
            init_status["warnings"].append("Ollama setup incomplete for local execution")
            init_status["warnings"].extend(ollama_status["suggestions"])
        
        # Set initialization flag
        _package_config["initialized"] = True
        
    except Exception as e:
        init_status["success"] = False
        init_status["errors"].append(f"Local package initialization failed: {e}")
    
    return init_status

def get_cache_manager() -> Optional['CacheManager']:
    """Get the package-level cache manager optimized for local use."""
    return _package_config.get("cache_manager")

def get_package_logger() -> Optional['EnhancedLogger']:
    """Get the package-level logger instance."""
    return _package_config.get("logger")

def create_local_assistant(config_path: str = "config.yaml",
                          data_dir: str = "data",
                          enable_caching: bool = True,
                          enable_reasoning: bool = True) -> 'DocumentAssistant':
    """Create a DocumentAssistant optimized for local Ollama execution."""
    # Initialize package if not already done
    if not _package_config["initialized"]:
        init_result = initialize_package(config_path, enable_caching, show_banner=True)
        
        # Display local setup status
        ollama_status = init_result.get("ollama_status", {})
        if not ollama_status.get("local_ready", False):
            print("\n⚠️  Local Setup Issues Detected:")
            for issue in ollama_status.get("issues", []):
                print(f"   • {issue}")
            print("\n💡 Quick Setup:")
            for suggestion in ollama_status.get("suggestions", [])[:3]:
                print(f"   • {suggestion}")
            print()
        else:
            system_info = init_result.get("local_info", {})
            memory_gb = system_info.get("memory", {}).get("total_gb", 0)
            print(f"✅ Local setup ready! ({memory_gb}GB RAM detected)")
            print()
    
    return DocumentAssistant(
        config_path=config_path,
        data_dir=data_dir,
        enable_caching=enable_caching,
        enable_reasoning=enable_reasoning
    )

# Alias for main function
create_assistant = create_local_assistant

class DocumentAssistant:
    """High-level interface for local AI Document Assistant with privacy-first design."""
    
    def __init__(self, 
                 config_path: str = "config.yaml",
                 data_dir: str = "data",
                 enable_caching: bool = True,
                 enable_reasoning: bool = True):
        """Initialize the Document Assistant for local execution."""
        self.config_path = config_path
        self.data_dir = data_dir
        self.enable_caching = enable_caching
        self.enable_reasoning = enable_reasoning
        
        # Get package-level components
        self.cache_manager = get_cache_manager()
        self.logger = get_package_logger()
        
        # Initialize core components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize components optimized for local execution."""
        self.embedding_model = get_embedding_model(
            config_path=self.config_path,
            cache_manager=self.cache_manager
        )
        
        self.index_manager = get_index_manager(
            cache_manager=self.cache_manager
        )
        
        self.llm = get_llm(
            config_path=self.config_path,
            cache_manager=self.cache_manager
        )
    
    def query(self, 
             question: str,
             format_type: str = "json",
             enable_reasoning: Optional[bool] = None) -> str:
        """Process a query using local models only."""
        reasoning_enabled = enable_reasoning if enable_reasoning is not None else self.enable_reasoning
        
        result = run_assistant(
            question,
            enable_reasoning_trace=reasoning_enabled,
            data_dir=self.data_dir,
            use_cache=self.enable_caching
        )
        
        # Format output if requested
        if format_type != "json":
            import json
            data = json.loads(result)
            
            if format_type == "text":
                return format_for_human(data)
            elif format_type == "markdown":
                return format_for_report(data)
            elif format_type == "executive":
                return format_for_business(data)
            elif format_type == "api":
                return format_for_api(data)
        
        return result
    
    def batch_query(self, 
                   questions: List[str],
                   enable_reasoning: bool = False) -> Dict[str, Any]:
        """Process multiple queries locally."""
        return run_batch_queries(
            questions,
            data_dir=self.data_dir,
            enable_reasoning=enable_reasoning,
            use_cache=self.enable_caching
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics for local execution."""
        stats = {}
        
        try:
            stats["embedding"] = self.embedding_model.get_embedding_stats()
        except Exception as e:
            stats["embedding"] = {"error": str(e)}
        
        try:
            stats["indexing"] = self.index_manager.get_performance_summary()
        except Exception as e:
            stats["indexing"] = {"error": str(e)}
        
        try:
            stats["llm"] = self.llm.get_usage_stats()
        except Exception as e:
            stats["llm"] = {"error": str(e)}
        
        if self.cache_manager:
            try:
                stats["cache"] = self.cache_manager.get_stats()
            except Exception as e:
                stats["cache"] = {"error": str(e)}
        
        # Add local execution metrics
        stats["local_execution"] = {
            "total_cost": "$0.00",
            "data_privacy": "100% local",
            "network_dependency": "none"
        }
        
        return stats

# Convenience functions for local execution
def quick_query(question: str, 
               data_dir: str = "data",
               format_type: str = "text") -> str:
    """Quick local query with automatic optimization."""
    assistant = create_local_assistant(data_dir=data_dir)
    return assistant.query(question, format_type=format_type)

def batch_analyze(questions: List[str], 
                 data_dir: str = "data") -> Dict[str, Any]:
    """Quick local batch analysis."""
    assistant = create_local_assistant(data_dir=data_dir)
    return assistant.batch_query(questions)

def validate_local_setup() -> bool:
    """Quick validation that everything is ready for local execution."""
    try:
        ollama_status = check_ollama_local_setup()
        return ollama_status.get("local_ready", False)
    except Exception:
        return False

# Package exports for local execution
__all__ = [
    # Core functions
    "run_assistant", "run_batch_queries",
    
    # Document processing
    "load_documents", "get_document_summary", "validate_documents",
    
    # Embedding system  
    "get_embedding_model", "embed_documents", "embed_query", "get_embedding_stats",
    
    # Index management
    "get_index_manager", "build_index", "search_index", "get_index_stats",
    
    # LLM interface
    "get_llm", "generate_response", "get_llm_stats",
    
    # Reasoning system
    "explain_reasoning", "detect_conflicts", "perform_multi_hop_reasoning",
    
    # Logging and monitoring
    "get_logger", "log", "log_performance", "get_performance_summary", "timer",
    
    # Caching system
    "CacheManager", "EmbeddingCache",
    
    # Output formatting
    "ResponseFormatter", "format_for_api", "format_for_human", "format_for_report", "format_for_business",
    
    # Local package management
    "initialize_package", "create_local_assistant", "create_assistant", "DocumentAssistant",
    
    # Local utilities
    "quick_query", "batch_analyze", "check_ollama_local_setup", "get_local_system_info", "recommend_ollama_models", "validate_local_setup",
    
    # Core classes
    "EnhancedEmbeddingModel", "IndexManager", "EnhancedLLM", "EnhancedLogger",
    
    # Package metadata
    "__version__", "__author__", "__description__"
]

# Auto-initialization for local execution
if not _package_config["initialized"]:
    try:
        # Silent auto-initialization optimized for local use
        initialize_package(show_banner=False)
    except Exception:
        # Silent failure - user will get guidance when they create an assistant
        pass
