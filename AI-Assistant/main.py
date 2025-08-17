#!/usr/bin/env python3
import sys
import json
import os
import requests
from typing import Dict, Any
from pathlib import Path
from core.rag_engine import run_assistant
from core.logger import log
from core.llm import check_ollama_status
from core.embedding import check_ollama_embeddings_status

def check_ollama_setup() -> Dict[str, Any]:
    """Comprehensive Ollama setup check."""
    status = {
        "ollama_running": False,
        "models_available": [],
        "embedding_models": [],
        "setup_complete": False,
        "issues": [],
        "suggestions": []
    }
    
    try:
        # Check if Ollama server is running
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            status["ollama_running"] = True
            models_data = response.json().get('models', [])
            status["models_available"] = [model['name'] for model in models_data]
            
            # Check for required models
            required_llm_models = ['llama3.2', 'llama3.1', 'llama2', 'mistral']
            required_embed_models = ['nomic-embed-text', 'all-minilm']
            
            available_llm = any(any(req in model for req in required_llm_models) 
                              for model in status["models_available"])
            
            status["embedding_models"] = [model for model in status["models_available"] 
                                        if any(embed in model.lower() for embed in ['embed', 'embedding'])]
            
            available_embed = len(status["embedding_models"]) > 0
            
            if not available_llm:
                status["issues"].append("No suitable LLM model found")
                status["suggestions"].append("Install a model: ollama pull llama3.2")
            
            if not available_embed:
                status["issues"].append("No embedding model found")
                status["suggestions"].append("Install embedding model: ollama pull nomic-embed-text")
            
            status["setup_complete"] = available_llm and available_embed
            
        else:
            status["issues"].append(f"Ollama server error: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        status["issues"].append("Cannot connect to Ollama server")
        status["suggestions"].extend([
            "1. Install Ollama: https://ollama.com/download",
            "2. Start Ollama: ollama serve",
            "3. Install models: ollama pull llama3.2 && ollama pull nomic-embed-text"
        ])
    except Exception as e:
        status["issues"].append(f"Unexpected error: {e}")
    
    return status

def validate_environment():
    """Validate environment setup for Ollama."""
    errors = []
    warnings = []
    
    # Check Ollama setup
    ollama_status = check_ollama_setup()
    
    if not ollama_status["ollama_running"]:
        errors.append("🔴 Ollama server is not running!")
        errors.extend([f"   {suggestion}" for suggestion in ollama_status["suggestions"]])
    elif not ollama_status["setup_complete"]:
        warnings.append("🟡 Ollama is running but setup incomplete:")
        warnings.extend([f"   {issue}" for issue in ollama_status["issues"]])
        warnings.extend([f"   {suggestion}" for suggestion in ollama_status["suggestions"]])
    else:
        print("✅ Ollama setup complete!")
        print(f"   Available LLM models: {[m for m in ollama_status['models_available'] if 'embed' not in m.lower()]}")
        print(f"   Available embedding models: {ollama_status['embedding_models']}")
    
    # Check data directory
    data_dir = Path("data")
    if not data_dir.exists():
        errors.append(f"🔴 Data directory '{data_dir}' not found!")
        return errors, warnings
    
    # Check for documents
    doc_files = list(data_dir.glob("*.pdf")) + list(data_dir.glob("*.txt"))
    if not doc_files:
        errors.append(f"🔴 No PDF or TXT files found in '{data_dir}'!")
    else:
        print(f"📄 Found {len(doc_files)} documents in data directory")
    
    return errors, warnings

def create_sample_data():
    """Create sample data and configuration for Ollama setup."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    sample_file = data_dir / "ollama_ml_guide.txt"
    sample_content = """# Local AI with Ollama - Machine Learning Guide

## Introduction to Local AI
Running AI models locally with Ollama provides several advantages:
- Complete privacy and data control
- No API costs or rate limits
- Offline functionality
- Custom model fine-tuning capabilities

## Ollama Supported Models

### Language Models
- **Llama 3.2**: Latest Meta model, excellent for general tasks
- **Llama 3.1**: Previous version, very capable for most use cases
- **Mistral**: Fast and efficient European model
- **DeepSeek Coder**: Specialized for code generation and analysis

### Embedding Models
- **nomic-embed-text**: High-quality text embeddings
- **all-minilm**: Lightweight embedding model
- **BGE models**: Bilingual and multilingual embeddings

## Local AI Benefits

### Privacy
All processing happens on your machine. No data leaves your system.

### Performance
- No network latency
- Consistent response times
- No rate limiting

### Cost Efficiency
- No per-token charges
- One-time hardware investment
- Unlimited usage

## Best Practices for Local AI

### Hardware Requirements
- **Minimum**: 8GB RAM, modern CPU
- **Recommended**: 16GB+ RAM, dedicated GPU
- **Optimal**: 32GB+ RAM, high-end GPU

### Model Selection
Choose models based on your hardware:
- Limited resources: Use smaller models (1B-3B parameters)
- Medium resources: Use 7B-8B parameter models
- High-end hardware: Use 13B+ parameter models

### Optimization Tips
1. Use appropriate model sizes for your hardware
2. Enable GPU acceleration when available
3. Optimize batch sizes for your memory
4. Use caching to avoid recomputation

## Document Analysis with Local AI
Local AI excels at document analysis because:
- Sensitive documents stay private
- No upload size limitations
- Consistent processing speed
- Custom prompt optimization
"""
    
    sample_file.write_text(sample_content, encoding='utf-8')
    print(f"✅ Created sample document: {sample_file}")
    
    # Create Ollama-specific config
    config_file = Path("config.yaml")
    if not config_file.exists():
        config_content = """# AI Document Assistant - Ollama Configuration

ollama:
  llm:
    provider: "ollama"
    model: "llama3.2"
    base_url: "http://localhost:11434"
    temperature: 0.1
    max_tokens: 1024
    max_retries: 3

  embedding:
    provider: "ollama"
    ollama_model: "nomic-embed-text"
    base_url: "http://localhost:11434"
    dimensions: 768

cache:
  enable_response_caching: true
  enable_embedding_cache: true
  cache_ttl: 3600

logging:
  level: "INFO"
  file: "outputs/activity.log"
"""
        config_file.write_text(config_content)
        print(f"✅ Created Ollama config: {config_file}")

def display_ollama_setup_instructions():
    """Display setup instructions for Ollama."""
    print("\n" + "="*60)
    print("🚀 OLLAMA SETUP INSTRUCTIONS")
    print("="*60)
    print("\n1. Install Ollama:")
    print("   Visit: https://ollama.com/download")
    print("   Or run: curl -fsSL https://ollama.com/install.sh | sh")
    
    print("\n2. Start Ollama server:")
    print("   ollama serve")
    
    print("\n3. Install required models:")
    print("   ollama pull llama3.2")
    print("   ollama pull nomic-embed-text")
    
    print("\n4. Optional: Install additional models:")
    print("   ollama pull llama3.2:1b    # Smaller, faster model")
    print("   ollama pull mistral        # Alternative LLM")
    
    print("\n5. Verify installation:")
    print("   ollama list")
    
    print("\n6. Test the setup:")
    print("   python main.py \"What are the benefits of local AI?\"")

def display_results(data):
    """Display results in a user-friendly format."""
    print("\n" + "="*60)
    print("🤖 AI DOCUMENT ASSISTANT RESPONSE (Ollama)")
    print("="*60)
    
    # Main answer
    print("\n📖 ANSWER:")
    print("-" * 40)
    print(data.get('answer', 'No answer provided'))
    
    # Sources
    sources = data.get('sources', [])
    if sources:
        print(f"\n📚 SOURCES ({len(sources)}):")
        for i, source in enumerate(sources, 1):
            print(f"  {i}. {source}")
    
    # Performance metrics
    metrics = data.get('performance_metrics', {})
    if metrics:
        print("\n⚡ PERFORMANCE:")
        proc_time = metrics.get('processing_time_seconds', 'N/A')
        docs_analyzed = metrics.get('documents_analyzed', 'N/A')
        print(f"  • Processing time: {proc_time} seconds")
        print(f"  • Documents analyzed: {docs_analyzed}")
        print(f"  • Model: Local Ollama")
    
    # Conflicts
    conflicts = data.get('conflicts_detected', [])
    if conflicts:
        print(f"\n⚠️ CONFLICTS DETECTED ({len(conflicts)}):")
        for i, conflict in enumerate(conflicts, 1):
            desc = conflict.get('description', 'Unknown conflict')
            severity = conflict.get('severity', 'unknown')
            print(f"  {i}. [{severity.upper()}] {desc}")
    
    # Reasoning trace (abbreviated for Ollama)
    reasoning = data.get('reasoning_trace', '')
    if reasoning and len(reasoning) > 100:
        print("\n🧠 REASONING PROCESS:")
        print("-" * 40)
        # Show first 300 characters for local models
        preview = reasoning[:300] + "..." if len(reasoning) > 300 else reasoning
        print(preview)

def main():
    """Enhanced main function with Ollama support."""
    print("🚀 AI Document Assistant (Ollama Edition)")
    print("=" * 45)
    
    try:
        # Validate environment
        errors, warnings = validate_environment()
        
        # Display warnings
        for warning in warnings:
            print(warning)
        
        # Handle errors
        if errors:
            print("\n🔴 Setup Issues Found:")
            for error in errors:
                print(error)
            
            print("\n🔧 Quick Fix:")
            print("Run this to create sample data:")
            print("  python -c \"from main import create_sample_data; create_sample_data()\"")
            
            if any("Ollama" in error for error in errors):
                display_ollama_setup_instructions()
            
            return 1
        
        # Get query
        if len(sys.argv) > 1:
            user_query = " ".join(sys.argv[1:])
        else:
            user_query = "What are the main benefits of using local AI models with Ollama?"
        
        print(f"❓ Query: {user_query}")
        print("\n🔄 Processing with local Ollama models... (this may take a moment)")
        
        # Run the assistant
        result = run_assistant(
            user_query, 
            enable_reasoning_trace=True,
            data_dir="data",
            use_cache=True
        )
        
        # Parse and display results
        try:
            response_data = json.loads(result)
            display_results(response_data)
            
        except json.JSONDecodeError as e:
            print(f"🔴 Error parsing response: {e}")
            print("Raw response:", result[:500] + "..." if len(result) > 500 else result)
            return 1
        
        print(f"\n💡 Try different queries by running:")
        print(f"  python main.py \"Your question here\"")
        print(f"  python cli.py interactive  # For interactive mode")
        print(f"\n🛠️ Model management:")
        print(f"  ollama list              # See installed models")
        print(f"  ollama pull <model>      # Install new models")
        print(f"  ollama rm <model>        # Remove models")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!")
        return 0
        
    except ImportError as e:
        print(f"🔴 Missing dependencies: {e}")
        print("Install with: pip install -r requirements.txt")
        return 1
        
    except Exception as e:
        print(f"🔴 Unexpected error: {e}")
        print(f"Error type: {type(e).__name__}")
        
        # Show helpful debugging info
        print(f"\n🔍 Debug Info:")
        print(f"  • Python version: {sys.version}")
        print(f"  • Current directory: {os.getcwd()}")
        print(f"  • Data directory exists: {Path('data').exists()}")
        
        # Check Ollama status for debugging
        try:
            ollama_status = check_ollama_setup()
            print(f"  • Ollama running: {ollama_status['ollama_running']}")
            print(f"  • Models available: {len(ollama_status['models_available'])}")
        except:
            print(f"  • Ollama status: Unknown")
        
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)