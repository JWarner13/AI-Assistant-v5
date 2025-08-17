import sys
import json
import argparse
import time
from pathlib import Path
from typing import List, Optional
from core.rag_engine import run_assistant, run_batch_queries
from core.llm import check_ollama_status
from core.embedding import check_ollama_embeddings_status
from core.logger import get_logger
from main import check_ollama_setup, display_ollama_setup_instructions

class InteractiveCLI:
    """Interactive command line interface for the AI assistant."""
    
    def __init__(self):
        self.logger = get_logger()
        self.session_queries = []
        
    def run_interactive(self):
        """Run interactive session."""
        print("🤖 AI Document Assistant - Interactive Mode")
        print("=" * 50)
        print("Type 'help' for commands, 'quit' to exit")
        
        # Check Ollama status
        self._check_system_status()
        
        while True:
            try:
                user_input = input("\n🔍 Query: ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                    
                if user_input.lower() == 'help':
                    self._show_help()
                    continue
                    
                if user_input.lower() == 'status':
                    self._show_status()
                    continue
                    
                if user_input.lower() == 'history':
                    self._show_history()
                    continue
                    
                if user_input.lower().startswith('batch'):
                    self._handle_batch_command(user_input)
                    continue
                
                # Process query
                self._process_query(user_input)
                self.session_queries.append(user_input)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def _check_system_status(self):
        """Check and display system status."""
        ollama_status = check_ollama_setup()
        
        if ollama_status["setup_complete"]:
            print("✅ System ready")
            print(f"   Models: {len(ollama_status['models_available'])} available")
        else:
            print("⚠️ System issues detected:")
            for issue in ollama_status["issues"]:
                print(f"   • {issue}")
            
            if not ollama_status["ollama_running"]:
                print("\n💡 Quick fix: Run 'ollama serve' in another terminal")
    
    def _process_query(self, query: str):
        """Process a single query and display results."""
        print("🔄 Processing...")
        start_time = time.time()
        
        try:
            result = run_assistant(
                query,
                enable_reasoning_trace=True,
                data_dir="data",
                use_cache=True
            )
            
            response_data = json.loads(result)
            processing_time = time.time() - start_time
            
            # Display answer
            print(f"\n📖 Answer:")
            print("-" * 40)
            print(response_data.get('answer', 'No answer provided'))
            
            # Display sources
            sources = response_data.get('sources', [])
            if sources:
                print(f"\n📚 Sources ({len(sources)}):")
                for i, source in enumerate(sources, 1):
                    print(f"  {i}. {source}")
            
            # Display performance
            print(f"\n⚡ Performance: {processing_time:.2f}s")
            
            # Display conflicts if any
            conflicts = response_data.get('conflicts_detected', [])
            if conflicts:
                print(f"\n⚠️ Conflicts detected: {len(conflicts)}")
                
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing response: {e}")
        except Exception as e:
            print(f"❌ Error processing query: {e}")
    
    def _show_help(self):
        """Show help information."""
        print("\n📖 Available Commands:")
        print("─" * 30)
        print("help         - Show this help")
        print("status       - Show system status")
        print("history      - Show query history")
        print("batch <file> - Process queries from file")
        print("quit/exit/q  - Exit interactive mode")
        print("\n💡 Tips:")
        print("• Ask questions about your documents")
        print("• Use complex queries for detailed analysis")
        print("• Check 'data/' folder for your documents")
    
    def _show_status(self):
        """Show detailed system status."""
        print("\n🔍 System Status:")
        print("─" * 20)
        
        # Ollama status
        ollama_status = check_ollama_setup()
        print(f"Ollama running: {'✅' if ollama_status['ollama_running'] else '❌'}")
        print(f"Models available: {len(ollama_status['models_available'])}")
        
        if ollama_status['models_available']:
            print("Available models:")
            for model in ollama_status['models_available']:
                print(f"  • {model}")
        
        # Data directory status
        data_dir = Path("data")
        if data_dir.exists():
            doc_files = list(data_dir.glob("*.pdf")) + list(data_dir.glob("*.txt"))
            print(f"Documents: {len(doc_files)} files in data/")
        else:
            print("Documents: ❌ data/ directory not found")
        
        # Session info
        print(f"Session queries: {len(self.session_queries)}")
    
    def _show_history(self):
        """Show query history for this session."""
        if not self.session_queries:
            print("No queries in this session yet.")
            return
            
        print(f"\n📝 Query History ({len(self.session_queries)}):")
        print("─" * 30)
        for i, query in enumerate(self.session_queries, 1):
            print(f"{i}. {query}")
    
    def _handle_batch_command(self, command: str):
        """Handle batch processing command."""
        parts = command.split()
        if len(parts) < 2:
            print("❌ Usage: batch <filename>")
            return
            
        filename = parts[1]
        if not Path(filename).exists():
            print(f"❌ File not found: {filename}")
            return
        
        try:
            with open(filename, 'r') as f:
                queries = [line.strip() for line in f if line.strip()]
            
            if not queries:
                print("❌ No queries found in file")
                return
                
            print(f"🔄 Processing {len(queries)} queries from {filename}...")
            
            results = run_batch_queries(queries, data_dir="data")
            
            # Display batch results
            print(f"\n📊 Batch Results:")
            print("─" * 20)
            summary = results.get('batch_summary', {})
            print(f"Total queries: {summary.get('total_queries', 0)}")
            print(f"Successful: {summary.get('successful_queries', 0)}")
            print(f"Failed: {summary.get('failed_queries', 0)}")
            print(f"Processing time: {summary.get('total_processing_time', 0):.2f}s")
            
        except Exception as e:
            print(f"❌ Error processing batch: {e}")

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AI Document Assistant CLI with Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s interactive              # Start interactive mode
  %(prog)s query "What is AI?"      # Single query
  %(prog)s batch queries.txt        # Batch processing
  %(prog)s status                   # Check system status
        """
    )
    
    parser.add_argument('mode', 
                       choices=['interactive', 'query', 'batch', 'status'],
                       help='Operation mode')
    
    parser.add_argument('input', 
                       nargs='?',
                       help='Query text or filename for batch mode')
    
    parser.add_argument('--data-dir', 
                       default='data',
                       help='Directory containing documents (default: data)')
    
    parser.add_argument('--no-reasoning', 
                       action='store_true',
                       help='Disable reasoning traces')
    
    parser.add_argument('--no-cache', 
                       action='store_true',
                       help='Disable caching')
    
    parser.add_argument('--format', 
                       choices=['json', 'text', 'markdown'],
                       default='text',
                       help='Output format (default: text)')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'interactive':
            cli = InteractiveCLI()
            cli.run_interactive()
            
        elif args.mode == 'query':
            if not args.input:
                parser.error("Query text required for query mode")
            
            result = run_assistant(
                args.input,
                enable_reasoning_trace=not args.no_reasoning,
                data_dir=args.data_dir,
                use_cache=not args.no_cache
            )
            
            if args.format == 'json':
                print(result)
            else:
                response_data = json.loads(result)
                print(f"Answer: {response_data.get('answer', 'No answer')}")
                
                sources = response_data.get('sources', [])
                if sources:
                    print(f"Sources: {', '.join(sources)}")
            
        elif args.mode == 'batch':
            if not args.input:
                parser.error("Filename required for batch mode")
                
            if not Path(args.input).exists():
                print(f"❌ File not found: {args.input}")
                return 1
            
            with open(args.input, 'r') as f:
                queries = [line.strip() for line in f if line.strip()]
            
            results = run_batch_queries(
                queries, 
                data_dir=args.data_dir,
                enable_reasoning=not args.no_reasoning,
                use_cache=not args.no_cache
            )
            
            if args.format == 'json':
                print(json.dumps(results, indent=2))
            else:
                summary = results.get('batch_summary', {})
                print(f"Processed {summary.get('total_queries', 0)} queries")
                print(f"Success rate: {summary.get('successful_queries', 0)}/{summary.get('total_queries', 0)}")
            
        elif args.mode == 'status':
            ollama_status = check_ollama_setup()
            
            print("🔍 System Status:")
            print(f"Ollama running: {'✅' if ollama_status['ollama_running'] else '❌'}")
            print(f"Models: {len(ollama_status['models_available'])}")
            
            if not ollama_status['setup_complete']:
                print("\n⚠️ Issues:")
                for issue in ollama_status['issues']:
                    print(f"  • {issue}")
                    
                print("\n💡 Suggestions:")
                for suggestion in ollama_status['suggestions']:
                    print(f"  • {suggestion}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())