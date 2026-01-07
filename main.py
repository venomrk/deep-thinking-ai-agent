"""
Deep Research Agent - Main Entry Point
Command-line interface for running research tasks.
"""

import asyncio
import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from deep_research_agent.core.orchestrator import Orchestrator
from deep_research_agent.core.base import OutputFormat
from deep_research_agent.config.settings import get_settings, configure, Settings


def print_banner():
    """Print welcome banner"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║          🔬 Deep Research Agent v1.0                          ║
║     Advanced AI-Powered Research & Verification System       ║
╚══════════════════════════════════════════════════════════════╝
    """)


async def run_research(query: str, output_format: str, output_file: str = None,
                       verbose: bool = False):
    """Run a research task"""
    # Map format string to enum
    format_map = {
        "report": OutputFormat.RESEARCH_REPORT,
        "summary": OutputFormat.EXECUTIVE_SUMMARY,
        "facts": OutputFormat.FACT_SHEET,
        "trace": OutputFormat.REASONING_TRACE,
        "json": OutputFormat.JSON
    }
    fmt = format_map.get(output_format, OutputFormat.RESEARCH_REPORT)
    
    # Setup LLM client
    llm_client = None
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    
    if api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            llm_client = genai.GenerativeModel("gemini-2.0-flash")
            if verbose:
                print("✓ Gemini API configured")
        except ImportError:
            print("⚠ google-generativeai not installed, running without LLM")
        except Exception as e:
            print(f"⚠ Could not configure Gemini: {e}")
    else:
        print("⚠ No API key found (set GEMINI_API_KEY), running without LLM")
    
    # Create orchestrator
    orchestrator = Orchestrator(llm_client=llm_client)
    
    # Progress callback
    def on_progress(session_id: str, progress: float, step: str):
        if verbose:
            bar = "█" * int(progress * 20) + "░" * (20 - int(progress * 20))
            print(f"\r[{bar}] {progress:.0%} - {step}", end="", flush=True)
    
    orchestrator.add_progress_callback(on_progress)
    
    # Initialize
    print(f"\n📋 Query: {query}")
    print(f"📄 Format: {output_format}")
    print(f"\n⏳ Starting research...\n")
    
    try:
        await orchestrator.initialize()
        
        # Run research
        start_time = datetime.now()
        result = await orchestrator.research(query, fmt)
        elapsed = (datetime.now() - start_time).total_seconds()
        
        print(f"\n\n✅ Research complete in {elapsed:.1f}s\n")
        print("=" * 60)
        
        # Output result
        if output_file:
            Path(output_file).write_text(result.content, encoding='utf-8')
            print(f"📁 Output saved to: {output_file}")
        else:
            print(result.content)
        
        print("=" * 60)
        print(f"\n📊 Summary:")
        print(f"   • Confidence: {result.confidence:.0%}")
        print(f"   • Sources: {len(result.sources)}")
        print(f"   • Claims: {len(result.claims)}")
        
        # Show stats
        if verbose:
            stats = orchestrator.get_reasoning_stats()
            print(f"\n🧠 Reasoning Stats:")
            print(f"   • Tree nodes: {stats.get('total_nodes', 0)}")
            print(f"   • Max depth: {stats.get('max_depth', 0)}")
            print(f"   • Explorations: {stats.get('explorations', 0)}")
        
        await orchestrator.shutdown()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


async def interactive_mode():
    """Run in interactive mode"""
    print_banner()
    print("Type 'quit' or 'exit' to stop, 'help' for commands.\n")
    
    settings = get_settings()
    orchestrator = Orchestrator()
    
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            llm_client = genai.GenerativeModel("gemini-2.0-flash")
            orchestrator.set_llm_client(llm_client)
            print("✓ LLM configured\n")
        except Exception:
            print("⚠ Running without LLM\n")
    
    await orchestrator.initialize()
    
    while True:
        try:
            query = input("\n🔍 Research query: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ('quit', 'exit', 'q'):
                print("Goodbye!")
                break
            
            if query.lower() == 'help':
                print("""
Commands:
  help     - Show this help
  stats    - Show memory stats
  clear    - Clear memory
  quit     - Exit

Just type your research question to start research.
                """)
                continue
            
            if query.lower() == 'stats':
                stats = orchestrator.get_memory_stats()
                print(f"\n📊 Memory Stats: {stats}")
                continue
            
            if query.lower() == 'clear':
                orchestrator.clear_memory()
                print("✓ Memory cleared")
                continue
            
            # Run research
            print()
            result = await orchestrator.research(query, OutputFormat.EXECUTIVE_SUMMARY)
            print("\n" + "=" * 60)
            print(result.content)
            print("=" * 60)
            print(f"Confidence: {result.confidence:.0%} | Sources: {len(result.sources)}")
            
        except KeyboardInterrupt:
            print("\nInterrupted. Type 'quit' to exit.")
        except Exception as e:
            print(f"Error: {e}")
    
    await orchestrator.shutdown()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Deep Research Agent - AI-powered research assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py "What is quantum computing?"
  python main.py "Compare Python and JavaScript" -f summary
  python main.py "AI trends in 2025" -o report.md -v
  python main.py --interactive
        """
    )
    
    parser.add_argument(
        "query",
        nargs="?",
        help="Research query (omit for interactive mode)"
    )
    
    parser.add_argument(
        "-f", "--format",
        choices=["report", "summary", "facts", "trace", "json"],
        default="report",
        help="Output format (default: report)"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Output file path"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="Deep Research Agent v1.0.0"
    )
    
    args = parser.parse_args()
    
    # Run appropriate mode
    if args.interactive or not args.query:
        asyncio.run(interactive_mode())
    else:
        exit_code = asyncio.run(run_research(
            args.query,
            args.format,
            args.output,
            args.verbose
        ))
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
