AI Document Assistant with Ollama
A sophisticated Retrieval-Augmented Generation (RAG) system that runs entirely locally using Ollama models. Provides advanced document analysis, reasoning capabilities, and structured responses while maintaining complete privacy.

🎯 Project Overview
These are the goals of this project:

✅ Goal 1: Document Processing & RAG
Document Parsing: PDF and TXT processing with intelligent chunking
Semantic Search: Vector similarity search using FAISS
LLM Summarization: Local model-powered content analysis
✅ Goal 2: LLM-Based Reasoning & Structured Responses
Complex Reasoning: Multi-document analysis with conflict detection
JSON Output: Structured responses for system integration
Reasoning Traces: Step-by-step explanation of analysis process
✅ Goal 3: System Optimization
Performance Optimization: Caching, batch processing, memory management
Local Execution: No API costs, complete privacy, offline functionality
Advanced Features: Conflict detection, multi-hop reasoning, performance monitoring
🚀 Quick Start
1. Install Ollama
bash
# Visit https://ollama.com/download or run:
curl -fsSL https://ollama.com/install.sh | sh
2. Start Ollama & Install Models
bash
# Start the server
ollama serve

# Install required models
ollama pull llama3.2          # Main LLM
ollama pull nomic-embed-text  # Embedding model
3. Install Python Dependencies
bash
pip install -r requirements.txt
4. Run the Assistant
bash
# Quick test
python main.py "What are the benefits of local AI?"

# Interactive mode
python cli.py interactive

# Batch processing
python cli.py batch queries.txt
📁 Project Structure
ai-document-assistant/
├── main.py                 # Main entry point
├── cli.py                  # Command line interface
├── config.yaml            # Configuration file
├── requirements.txt        # Python dependencies
├── data/                   # Document directory
├── outputs/               # Logs and results
├── core/                  # Core modules
│   ├── __init__.py        # Package initialization
│   ├── cache.py           # Caching system
│   ├── document_loader.py # Document processing
│   ├── embedding.py       # Embedding models
│   ├── index.py           # Vector indexing
│   ├── llm.py             # Language models
│   ├── logger.py          # Logging system
│   ├── output_formatter.py # Response formatting
│   ├── rag_engine.py      # Main RAG pipeline
│   └── reasoning.py       # Advanced reasoning
└── README.md              # This file
🔧 Configuration
Model Selection (config.yaml)
yaml
ollama:
  llm:
    model: "llama3.2"        # Main language model
    base_url: "http://localhost:11434"
    temperature: 0.1
    max_tokens: 1024
  
  embedding:
    provider: "ollama"
    ollama_model: "nomic-embed-text"
    dimensions: 768
Hardware Recommendations
Setup	RAM	Model	Performance
Minimum	8GB	llama3.2:1b	Fast, Good quality
Recommended	16GB	llama3.2	Balanced
High-end	32GB+	llama3.1:8b	Best quality
💻 Usage Examples
Basic Document Analysis
bash
python main.py "What are the main topics in these documents?"
Complex Reasoning
bash
python main.py "Compare the different approaches and identify conflicts"
Interactive Mode
bash
python cli.py interactive
Batch Processing
bash
# Create queries.txt with one query per line
echo "What is machine learning?" > queries.txt
echo "What are the benefits of AI?" >> queries.txt
python cli.py batch queries.txt
Structured JSON Output
python
from core.rag_engine import run_assistant
import json

result = run_assistant("Analyze the key findings")
data = json.loads(result)
print(json.dumps(data, indent=2))
🎛️ Advanced Features
1. Conflict Detection
Automatically identifies contradictions between sources:

json
{
  "conflicts_detected": [
    {
      "type": "contradiction",
      "description": "Sources disagree on implementation approach",
      "sources_involved": ["doc1.pdf", "doc2.pdf"],
      "severity": "medium"
    }
  ]
}
2. Reasoning Traces
Step-by-step explanation of analysis:

json
{
  "reasoning_trace": "## Reasoning Process\n### Step 1: Information Gathering\n[Detailed reasoning steps...]"
}
3. Performance Monitoring
bash
# Check system performance
python cli.py status

# View detailed logs
tail -f outputs/activity.log
4. Multiple Output Formats
python
from core.output_formatter import format_for_api, format_for_human

# API-optimized JSON
api_response = format_for_api(response_data)

# Human-readable text
text_response = format_for_human(response_data)
🛠️ Model Management
Install Additional Models
bash
# Smaller model for limited hardware
ollama pull llama3.2:1b

# Alternative models
ollama pull mistral
ollama pull deepseek-coder  # For code analysis

# Check installed models
ollama list
Model Switching
Update config.yaml:

yaml
ollama:
  llm:
    model: "llama3.2:1b"  # Switch to smaller model
🔍 Troubleshooting
Common Issues
"Cannot connect to Ollama server"
bash
# Check if Ollama is running
ollama list

# Start Ollama if not running
ollama serve
"Model not found"
bash
# Install missing models
ollama pull llama3.2
ollama pull nomic-embed-text
Slow Performance
yaml
# Use smaller model in config.yaml
ollama:
  llm:
    model: "llama3.2:1b"
    max_tokens: 512
Out of Memory
Use smaller models (1b-3b parameters)
Reduce batch sizes in config
Close other applications
Debug Information
bash
# System status
python cli.py status

# Detailed logs
python main.py "test query" 2>&1 | tee debug.log

# Check Ollama directly
curl http://localhost:11434/api/tags
📊 Performance Benchmarks
Model	RAM Usage	Speed	Quality	Use Case
llama3.2:1b	~2GB	Fast	Good	Development/Testing
llama3.2	~4GB	Medium	Very Good	General Use
llama3.1:8b	~8GB	Slower	Excellent	High Quality Analysis
🔒 Privacy & Security
Complete Privacy: All processing happens locally
No Data Transmission: Documents never leave your machine
Offline Functionality: Works without internet connection
No API Keys: No external service dependencies
🎯 Project Features Demonstrated
Document Processing & RAG ✅
 Efficient PDF/TXT parsing with metadata extraction
 Intelligent text chunking with overlap management
 Semantic vector search using FAISS
 Local embedding generation with Ollama
LLM Reasoning & Structured Output ✅
 Complex multi-document reasoning
 JSON-structured responses for system integration
 Reasoning trace generation and explanation
 Ambiguous query handling with confidence metrics
System Optimization ✅
 Multi-level caching (embeddings, responses, search results)
 Batch processing for efficiency
 Performance monitoring and logging
 Conflict detection and resolution
 Memory-efficient processing
🚀 Future Enhancements
 Web interface for document upload
 Custom model fine-tuning
 Integration with more document formats
 Advanced visualization of reasoning chains
 Multi-language support
📝 Contributing
Fork the repository
Create a feature branch
Make your changes
Test with local Ollama setup
Submit a pull request
📄 License
MIT License - See LICENSE file for details

🆘 Support
For issues and questions:

Check this README's troubleshooting section
Verify Ollama installation: ollama --version
Check model availability: ollama list
Review logs in outputs/activity.log
Open an issue with system details and error logs


