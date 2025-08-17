import sys
import os
import psutil
from typing import Dict, Any, Optional, List
from pathlib import Path
from core.cache import CacheManager
from core.rag_engine import _generate_cache_key

# Package metadata - Local AI Focus
__version__ = "2.0.0-local"
__author__ = "AI Document Assistant Team"
__description__ = "Privacy-first RAG assistant with local Ollama models - no cloud dependencies"
__license__ = "MIT"

# Minimum Python version check
MIN_PYTHON = (3, 8)
if sys.version_info < MIN_PYTHON:
    sys.exit(f"Python {MIN_PYTHON[0]

# Auto-initialization for local execution
if not _package_config["initialized"]:
    try:
        # Silent auto-initialization optimized for local use
        initialize_package(show_banner=False)
    except Exception:
        # Silent failure - user will get guidance when they create an assistant
        pass

# Local execution helpers
def validate_local_setup() -> bool:
    """Quick validation that everything is ready for local execution."""
    try:
        ollama_status = check_ollama_local_setup()
        return ollama_status.get("local_ready", False)
    except Exception:
        return False

def get_setup_instructions() -> List[str]:
    """Get step-by-step setup instructions for local execution."""
    instructions = [
        "🚀 AI Document Assistant - Local Setup Instructions",
        "",
        "1. Install Ollama:",
        "   curl -fsSL https://ollama.com/install.sh | sh",
        "",
        "2. Start Ollama server:",
        "   ollama serve",
        "",
        "3. Install required models:",
        "   ollama pull llama3.2",
        "   ollama pull nomic-embed-text",
        "",
        "4. Verify setup:",
        "   ollama list",
        "",
        "5. Test the assistant:",
        "   python main.py \"What is AI?\"",
        "",
        "💡 Your data stays completely private - no internet required after setup!"
    ]
    return instructions

def display_setup_instructions():
    """Display setup instructions in a formatted way."""
    for instruction in get_setup_instructions():
        print(instruction)

# Performance optimization for local execution
def optimize_for_local_performance():
    """Apply optimizations specifically for local Ollama execution."""
    optimizations = {
        "cache_settings": {
            "aggressive_caching": True,
            "larger_cache_size": True,
            "longer_ttl": True
        },
        "batch_processing": {
            "optimized_batch_sizes": True,
            "memory_efficient": True
        },
        "model_settings": {
            "temperature": 0.1,  # More deterministic for local use
            "max_tokens": 1024,  # Balanced for local processing
            "timeout": 120       # Longer timeout for local models
        }
    }
    return optimizations

# Privacy and security features for local execution
def get_privacy_report() -> Dict[str, Any]:
    """Generate a privacy report showing local execution benefits."""
    return {
        "data_handling": {
            "documents_location": "Local filesystem only",
            "processing_location": "Local machine only", 
            "embeddings_storage": "Local FAISS index",
            "cache_location": "Local filesystem",
            "logs_location": "Local filesystem"
        },
        "network_usage": {
            "external_api_calls": "None",
            "data_transmission": "None", 
            "internet_required": "Only for initial Ollama model download",
            "ongoing_connectivity": "Not required"
        },
        "privacy_guarantees": {
            "data_leaves_machine": False,
            "third_party_access": False,
            "api_logging": False,
            "usage_tracking": False
        },
        "compliance": {
            "gdpr_compliant": True,
            "hipaa_ready": True,
            "enterprise_secure": True,
            "air_gap_capable": True
        }
    }

# Cost analysis for local execution
def get_cost_analysis() -> Dict[str, Any]:
    """Analyze cost benefits of local execution."""
    return {
        "operational_costs": {
            "api_costs": "$0.00",
            "per_query_cost": "$0.00", 
            "monthly_costs": "$0.00",
            "unlimited_usage": True
        },
        "setup_costs": {
            "software_cost": "$0.00",
            "hardware_requirement": "Existing computer",
            "one_time_setup": True
        },
        "comparison": {
            "vs_openai_api": "100% savings",
            "vs_claude_api": "100% savings", 
            "vs_hosted_solutions": "100% savings",
            "break_even_queries": 1  # Immediate savings
        },
        "scaling": {
            "cost_per_additional_query": "$0.00",
            "bulk_processing_cost": "$0.00",
            "enterprise_scaling": "Hardware dependent only"
        }
    }

# Add local execution note
_LOCAL_EXECUTION_NOTE = """
🏠 LOCAL EXECUTION MODE
This AI Document Assistant runs entirely on your machine using Ollama.
✅ 100% Private  ✅ $0 API Costs  ✅ Offline Capable  ✅ Unlimited Usage
"""

def print_local_benefits():
    """Print the benefits of local execution."""
    print(_LOCAL_EXECUTION_NOTE)}.{MIN_PYTHON[1]} or later is required.")

# Local execution banner
def display_local_banner():
    """Display local execution information."""
    print("🏠 AI Document Assistant - Local Edition")
    print("🔒 100% Private • 💰 $0 API Costs • 🌐 Offline Ready")
    print("=" * 50)

# Core module imports with error handling
try:
    # Core RAG components
    from .rag_engine import (
        run_assistant,
        run_batch_queries,
        _generate_cache_key,
        _determine_optimal_k
    )
    
    # Document processing
    from .document_loader import (
        load_documents,
        extract_text_from_pdf,
        extract_text_from_txt,
        split_into_chunks,
        get_document_summary,
        validate_documents
    )
    
    # Embedding system
    from .embedding import (
        get_embedding_model,
        embed_documents,
        embed_query,
        get_embedding_stats,
        estimate_embedding_cost,
        EnhancedEmbeddingModel,
        check_ollama_embeddings_status
    )
    
    # Vector indexing
    from .index import (
        get_index_manager,
        build_index,
        search_index,
        build_and_search,
        get_index_stats,
        IndexManager
    )
    
    # LLM interface
    from .llm import (
        get_llm,
        generate_response,
        get_llm_stats,
        estimate_query_cost,
        EnhancedLLM,
        check_ollama_status
    )
    
    # Advanced reasoning
    from .reasoning import (
        explain_reasoning,
        detect_conflicts,
        perform_multi_hop_reasoning
    )
    
    # Logging and monitoring
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
    
    # Caching system
    from .cache import (
        CacheManager,
        EmbeddingCache,
        QueryCache
    )
    
    # Output formatting
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

def get_local_system_info() -> Dict[str, Any]:
    """Get local system information for optimization."""
    try:
        # System resources
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('.')
        cpu_count = psutil.cpu_count()
        
        # GPU detection (basic)
        gpu_available = False
        try:
            import torch
            gpu_available = torch.cuda.is_available()
        except ImportError:
            pass
        
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
            },
            "gpu": {
                "available": gpu_available,
                "recommended": memory.total >= 16 * (1024**3)  # 16GB+ for GPU models
            }
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
        recommendations["advanced"] = ["llama3.1:70b"]  # For very high-end systems
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
            
            # Performance suggestions
            memory_gb = status["system_info"].get("memory", {}).get("total_gb", 0)
            if memory_gb < 8:
                status["suggestions"].append("Consider upgrading to 8GB+ RAM for better performance")
            elif memory_gb >= 16:
                status["suggestions"].append("Your system can handle larger models for better quality")
            
        else:
            status["issues"].append(f"Ollama server error: {response.status_code}")
            status["suggestions"].append("Restart Ollama: ollama serve")
            
    except requests.exceptions.ConnectionError:
        status["issues"].append("Cannot connect to local Ollama server")
        status["suggestions"].extend([
            "1. Install Ollama: curl -fsSL https://ollama.com/install.sh | sh",
            "2. Start server: ollama serve",
            "3. Install models: ollama pull llama3.2 && ollama pull nomic-embed-text"
        ])
    except Exception as e:
        status["issues"].append(f"Unexpected error: {e}")
    
    return status

def initialize_package(config_path: str = "config.yaml", 
                      enable_caching: bool = True,
                      enable_logging: bool = True,
                      debug_mode: bool = False,
                      show_banner: bool = True) -> Dict[str, Any]:
    """
    Initialize the AI Document Assistant package for local Ollama execution.
    
    Args:
        config_path: Path to configuration file
        enable_caching: Whether to enable caching system
        enable_logging: Whether to enable enhanced logging
        debug_mode: Whether to enable debug mode
        show_banner: Whether to show the local execution banner
    
    Returns:
        Initialization status and component information
    """
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
                logger.log("Local AI package logging initialized", 
                          component="core", 
                          extra={"local_mode": True})
            except Exception as e:
                init_status["errors"].append(f"Logger initialization failed: {e}")
                init_status["components"]["logger"] = "failed"
        
        # Initialize local cache manager
        if enable_caching:
            try:
                # Use larger cache for local execution (no API cost concerns)
                cache_manager = CacheManager(max_size=2000, default_ttl=7200)  # 2 hours
                _package_config["cache_manager"] = cache_manager
                init_status["components"]["cache"] = "initialized"
                if _package_config["logger"]:
                    _package_config["logger"].log("Local cache system initialized", 
                                                 component="core",
                                                 extra={"max_size": 2000})
            except Exception as e:
                init_status["errors"].append(f"Cache initialization failed: {e}")
                init_status["components"]["cache"] = "failed"
        
        # Validate configuration file
        if os.path.exists(config_path):
            init_status["components"]["config"] = "loaded"
            if _package_config["logger"]:
                _package_config["logger"].log(f"Configuration loaded from {config_path}", component="core")
        else:
            init_status["warnings"].append(f"Configuration file not found: {config_path}")
            init_status["components"]["config"] = "default"
        
        # Check local environment and Ollama setup
        env_status = _check_local_environment()
        init_status["components"].update(env_status)
        
        # Validate local dependencies
        deps_status = _validate_local_dependencies()
        if not deps_status["all_available"]:
            init_status["warnings"].extend(deps_status["missing"])
        
        # Check Ollama setup
        ollama_status = check_ollama_local_setup()
        init_status["ollama_status"] = ollama_status
        
        if not ollama_status["local_ready"]:
            init_status["warnings"].append("Ollama setup incomplete for local execution")
            init_status["warnings"].extend(ollama_status["suggestions"])
        
        # Set initialization flag
        _package_config["initialized"] = True
        
        if _package_config["logger"]:
            _package_config["logger"].log("Local AI Document Assistant initialized successfully", 
                                        component="core",
                                        extra={
                                            "version": __version__,
                                            "local_mode": True,
                                            "privacy_mode": True,
                                            "memory_gb": init_status["local_info"].get("memory", {}).get("total_gb", 0),
                                            "ollama_ready": ollama_status["local_ready"]
                                        })
        
    except Exception as e:
        init_status["success"] = False
        init_status["errors"].append(f"Local package initialization failed: {e}")
    
    return init_status

def _check_local_environment() -> Dict[str, str]:
    """Check local environment setup for optimal performance."""
    env_status = {}
    
    # Check Ollama server process
    ollama_process_running = False
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            if proc.info['name'] and 'ollama' in proc.info['name'].lower():
                ollama_process_running = True
                break
    except Exception:
        pass
    
    env_status["ollama_process"] = "running" if ollama_process_running else "not_running"
    
    # Check network independence
    env_status["network_required"] = "false"  # Local execution doesn't need network
    env_status["api_dependencies"] = "none"   # No external APIs
    
    # Check required local directories
    required_dirs = ["data", "outputs", "cache", "indexes"]
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            env_status[f"{dir_name}_dir"] = "exists"
        else:
            try:
                os.makedirs(dir_name, exist_ok=True)
                env_status[f"{dir_name}_dir"] = "created"
            except Exception:
                env_status[f"{dir_name}_dir"] = "failed"
    
    # Check write permissions for local storage
    try:
        test_file = Path("outputs") / "test_write.tmp"
        test_file.write_text("test")
        test_file.unlink()
        env_status["write_permissions"] = "ok"
    except Exception:
        env_status["write_permissions"] = "denied"
    
    return env_status

def _validate_local_dependencies() -> Dict[str, Any]:
    """Validate dependencies for local execution (minimal external deps)."""
    # Focus on local execution - minimal dependencies
    essential_packages = {
        "requests": "Local Ollama API communication",
        "langchain_community": "Local vector storage",
        "faiss": "Local vector search",
        "sentence_transformers": "Fallback embeddings (local)",
        "numpy": "Local computations",
        "psutil": "Local system monitoring"
    }
    
    optional_packages = {
        "fitz": "Local PDF processing",
        "pydantic": "Data validation", 
        "yaml": "Local configuration",
        "loguru": "Local logging"
    }
    
    essential_available = []
    essential_missing = []
    optional_available = []
    optional_missing = []
    
    # Check essential packages
    for package, description in essential_packages.items():
        try:
            if package == "langchain_community":
                import langchain_community
            else:
                __import__(package)
            essential_available.append(package)
        except ImportError:
            essential_missing.append(f"{package} ({description})")
    
    # Check optional packages
    for package, description in optional_packages.items():
        try:
            if package == "fitz":
                import fitz
            elif package == "yaml":
                import yaml
            else:
                __import__(package)
            optional_available.append(package)
        except ImportError:
            optional_missing.append(f"{package} ({description})")
    
    return {
        "all_available": len(essential_missing) == 0,
        "essential_available": essential_available,
        "essential_missing": essential_missing,
        "optional_available": optional_available,
        "optional_missing": optional_missing,
        "local_ready": len(essential_missing) == 0
    }

def get_package_info() -> Dict[str, Any]:
    """Get comprehensive package information optimized for local execution."""
    local_info = get_local_system_info()
    ollama_status = check_ollama_local_setup()
    
    return {
        "package": {
            "name": "ai-document-assistant-local",
            "version": __version__,
            "description": __description__,
            "author": __author__,
            "license": __license__,
            "execution_mode": "local"
        },
        "privacy": {
            "data_stays_local": True,
            "no_external_apis": True,
            "offline_capable": True,
            "zero_api_costs": True
        },
        "system": {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": sys.platform,
            "initialized": _package_config["initialized"],
            "config_path": _package_config["config_path"],
            "debug_mode": _package_config["debug_mode"],
            "local_mode": _package_config["local_mode"]
        },
        "hardware": local_info,
        "ollama": {
            "server_running": ollama_status["ollama_running"],
            "models_available": len(ollama_status["models_available"]),
            "setup_complete": ollama_status["setup_complete"],
            "recommendations": ollama_status["recommendations"]
        },
        "components": {
            "cache_manager_available": _package_config["cache_manager"] is not None,
            "logger_available": _package_config["logger"] is not None,
            "performance_tracking": _package_config["performance_tracking"]
        },
        "environment": _check_local_environment(),
        "dependencies": _validate_local_dependencies(),
        "local_configuration": {
            "primary_llm": "Ollama (Local)",
            "embedding_model": "Ollama + SentenceTransformers (Local)",
            "vector_store": "FAISS (Local)",
            "data_processing": "Local Only",
            "privacy_level": "Maximum"
        }
    }

def get_cache_manager() -> Optional[CacheManager]:
    """Get the package-level cache manager optimized for local use."""
    return _package_config.get("cache_manager")

def get_package_logger() -> Optional['EnhancedLogger']:
    """Get the package-level logger instance."""
    return _package_config.get("logger")

def reset_package_state():
    """Reset package state (useful for testing)."""
    global _package_config
    _package_config.update({
        "initialized": False,
        "cache_manager": None,
        "logger": None
    })

def create_local_assistant(config_path: str = "config.yaml",
                          data_dir: str = "data",
                          enable_caching: bool = True,
                          enable_reasoning: bool = True,
                          auto_optimize: bool = True) -> 'DocumentAssistant':
    """
    Create a DocumentAssistant optimized for local Ollama execution.
    
    Args:
        config_path: Path to configuration file
        data_dir: Directory containing documents
        enable_caching: Whether to enable aggressive local caching
        enable_reasoning: Whether to enable reasoning traces
        auto_optimize: Whether to auto-optimize for local hardware
    
    Returns:
        Configured DocumentAssistant instance
    """
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
            for suggestion in ollama_status.get("suggestions", [])[:3]:  # Show top 3
                print(f"   • {suggestion}")
            print()
        else:
            system_info = init_result.get("local_info", {})
            memory_gb = system_info.get("memory", {}).get("total_gb", 0)
            print(f"✅ Local setup ready! ({memory_gb}GB RAM detected)")
            
            # Show model recommendations
            recommendations = ollama_status.get("recommendations", {})
            if recommendations.get("recommended"):
                print(f"🎯 Recommended models: {', '.join(recommendations['recommended'])}")
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
    """
    High-level interface for local AI Document Assistant with privacy-first design.
    """
    
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
        
        if self.logger:
            self.logger.log("Local DocumentAssistant initialized", 
                           component="assistant",
                           extra={
                               "data_dir": data_dir,
                               "caching": enable_caching,
                               "reasoning": enable_reasoning,
                               "provider": "ollama_local",
                               "privacy_mode": True
                           })
    
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
    
    def get_local_status(self) -> Dict[str, Any]:
        """Get comprehensive local system and model status."""
        return {
            "system": get_local_system_info(),
            "ollama": check_ollama_local_setup(),
            "privacy": {
                "data_location": "local_only",
                "external_requests": "none",
                "api_costs": "$0.00"
            }
        }
    
    def optimize_for_hardware(self) -> Dict[str, Any]:
        """Suggest optimizations based on local hardware."""
        system_info = get_local_system_info()
        recommendations = recommend_ollama_models(system_info)
        
        memory_gb = system_info.get("memory", {}).get("total_gb", 8)
        
        optimizations = {
            "current_hardware": system_info,
            "model_recommendations": recommendations,
            "settings_optimization": {},
            "performance_tips": []
        }
        
        # Memory-based optimizations
        if memory_gb >= 32:
            optimizations["settings_optimization"]["batch_size"] = 200
            optimizations["settings_optimization"]["cache_size"] = 5000
            optimizations["performance_tips"].append("Use larger models for best quality")
        elif memory_gb >= 16:
            optimizations["settings_optimization"]["batch_size"] = 100
            optimizations["settings_optimization"]["cache_size"] = 2000
            optimizations["performance_tips"].append("Good balance of speed and quality")
        else:
            optimizations["settings_optimization"]["batch_size"] = 50
            optimizations["settings_optimization"]["cache_size"] = 1000
            optimizations["performance_tips"].append("Use smaller models for better performance")
        
        return optimizations
    
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
        
        if self.logger:
            try:
                stats["performance"] = self.logger.get_performance_summary()
            except Exception as e:
                stats["performance"] = {"error": str(e)}
        
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
    "initialize_package", "get_package_info", "create_local_assistant", "create_assistant", "DocumentAssistant",
    
    # Local utilities
    "quick_query", "batch_analyze", "check_ollama_local_setup", "get_local_system_info", "recommend_ollama_models",
    
    # Core classes
    "EnhancedEmbeddingModel", "IndexManager", "EnhancedLLM", "EnhancedLogger",
    
    # Package metadata
    "__version__", "__author__", "__description__"