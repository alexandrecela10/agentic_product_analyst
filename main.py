"""
Product Success Tracking Agent - Main Entry Point

This module provides multiple ways to run the agent:
1. FastAPI server (REST API)
2. MCP server (for AI assistants)
3. CLI for quick testing
"""

import argparse
import uvicorn
import asyncio
from src.core.observability import init_observability, get_logger


def run_api(host: str = "0.0.0.0", port: int = 8000):
    """
    Run the FastAPI server.
    
    Provides REST API endpoints for:
    - /pipeline/run - Full pipeline execution
    - /rag/index - Index knowledge base
    - /database/explore - Explore tables
    - /grain/detect - Detect grain
    """
    from src.api.routes import app
    
    init_observability()
    logger = get_logger(__name__)
    logger.info("starting_api_server", host=host, port=port)
    
    uvicorn.run(app, host=host, port=port)


def run_mcp():
    """
    Run the MCP server.
    
    Exposes tools via Model Context Protocol for AI assistants.
    """
    from src.mcp.server import main as mcp_main
    
    asyncio.run(mcp_main())


def run_cli(feature_name: str, additional_context: str = ""):
    """
    Run the pipeline from command line.
    
    Quick way to test the agent without starting a server.
    """
    from src.orchestrator.pipeline import run_pipeline
    import json
    
    init_observability()
    logger = get_logger(__name__)
    logger.info("running_cli", feature=feature_name)
    
    result = run_pipeline(feature_name, additional_context)
    
    # Pretty print the result
    print("\n" + "="*60)
    print("PRODUCT SUCCESS TRACKING AGENT - RESULTS")
    print("="*60)
    
    print(f"\n📋 Feature: {result.feature_context.name}")
    print(f"📝 Description: {result.feature_context.description}")
    print(f"🎯 Purpose: {result.feature_context.purpose}")
    print(f"🔒 Confidence: {result.feature_context.confidence:.0%}")
    
    print(f"\n📊 Primary Metrics ({len(result.success_framework.primary_metrics)}):")
    for m in result.success_framework.primary_metrics:
        print(f"   - {m.display_name}: {m.description}")
    
    print(f"\n📈 Secondary Metrics ({len(result.success_framework.secondary_metrics)}):")
    for m in result.success_framework.secondary_metrics:
        print(f"   - {m.display_name}")
    
    print(f"\n🗄️ Eligible Tables ({len(result.eligible_tables)}):")
    for t in result.eligible_tables:
        grain = result.grain_results.get(t.table.name)
        grain_str = f" (grain: {grain.primary_grain.value})" if grain else ""
        print(f"   - {t.table.name}: score={t.score:.2f}{grain_str}")
    
    print(f"\n⚡ LLM Calls: {result.llm_calls}")
    print(f"⏱️ Execution Time: {result.execution_time_ms:.0f}ms")
    
    if result.errors:
        print(f"\n⚠️ Errors ({len(result.errors)}):")
        for e in result.errors:
            print(f"   - {e}")
    
    print("\n" + "="*60)
    
    # Also output JSON for programmatic use
    print("\n📄 Full JSON output saved to: pipeline_result.json")
    with open("pipeline_result.json", "w") as f:
        json.dump(result.model_dump(), f, indent=2, default=str)


def main():
    """
    Main entry point with argument parsing.
    """
    parser = argparse.ArgumentParser(
        description="Product Success Tracking Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run FastAPI server
  python main.py api
  
  # Run MCP server
  python main.py mcp
  
  # Run CLI with a feature name
  python main.py cli "AI Search"
  
  # Run CLI with additional context
  python main.py cli "AI Search" --context "Focus on conversion metrics"
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # API command
    api_parser = subparsers.add_parser("api", help="Run FastAPI server")
    api_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    api_parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    
    # MCP command
    subparsers.add_parser("mcp", help="Run MCP server")
    
    # CLI command
    cli_parser = subparsers.add_parser("cli", help="Run pipeline from CLI")
    cli_parser.add_argument("feature", help="Feature name to analyze")
    cli_parser.add_argument("--context", default="", help="Additional context")
    
    args = parser.parse_args()
    
    if args.command == "api":
        run_api(args.host, args.port)
    elif args.command == "mcp":
        run_mcp()
    elif args.command == "cli":
        run_cli(args.feature, args.context)
    else:
        # Default: show help
        parser.print_help()


if __name__ == "__main__":
    main()
