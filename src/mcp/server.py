"""
MCP Server for the Product Success Tracking Agent.

Exposes tools via the Model Context Protocol (MCP) for use by
AI assistants like Claude, Cursor, etc.

Why MCP?
- Emerging standard for AI tool integration
- Any MCP-compatible client can use these tools
- Enables composability with other agents
- Hands-on experience with production patterns
"""

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import json
from typing import Any

from src.orchestrator.pipeline import ProductSuccessPipeline
from src.rag.indexer import index_knowledge_base
from src.deterministic.database_explorer import explore_database, find_eligible_tables
from src.deterministic.grain_detector import detect_grain_from_model
from src.core.models import (
    PipelineInput,
    TableMetadataModel,
    ColumnInfoModel,
    SemanticType
)
from src.core.observability import init_observability, get_logger

logger = get_logger(__name__)

# Create MCP server
server = Server("product-success-agent")


# --- Tool Definitions ---

@server.list_tools()
async def list_tools() -> list[Tool]:
    """
    List all available tools.
    
    Each tool corresponds to a capability of the Product Success Agent.
    """
    return [
        Tool(
            name="discover_feature_context",
            description="Discover context about a product feature using RAG. Searches company docs and extracts purpose, target users, success criteria, and recommended metrics.",
            inputSchema={
                "type": "object",
                "properties": {
                    "feature_name": {
                        "type": "string",
                        "description": "Name of the feature to discover (e.g., 'AI Search')"
                    },
                    "additional_context": {
                        "type": "string",
                        "description": "Optional additional context from the user"
                    }
                },
                "required": ["feature_name"]
            }
        ),
        Tool(
            name="run_full_pipeline",
            description="Run the complete Product Success Tracking pipeline. Discovers context, generates success framework, finds eligible tables, and detects grain.",
            inputSchema={
                "type": "object",
                "properties": {
                    "feature_name": {
                        "type": "string",
                        "description": "Name of the feature to analyze"
                    },
                    "additional_context": {
                        "type": "string",
                        "description": "Optional additional context"
                    },
                    "skip_indexing": {
                        "type": "boolean",
                        "description": "Skip re-indexing knowledge base (faster if already indexed)"
                    }
                },
                "required": ["feature_name"]
            }
        ),
        Tool(
            name="explore_database",
            description="Explore all tables in the data warehouse. Returns metadata including columns, types, cardinality, and semantic types.",
            inputSchema={
                "type": "object",
                "properties": {
                    "data_directory": {
                        "type": "string",
                        "description": "Path to data directory (default: ./data/mock_warehouse)"
                    }
                }
            }
        ),
        Tool(
            name="find_eligible_tables",
            description="Find tables relevant to a feature based on keywords. Scores tables by column match, required columns, freshness, and row count.",
            inputSchema={
                "type": "object",
                "properties": {
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Keywords to match against table columns"
                    },
                    "data_directory": {
                        "type": "string",
                        "description": "Path to data directory"
                    },
                    "min_score": {
                        "type": "number",
                        "description": "Minimum relevance score (0-1)"
                    }
                },
                "required": ["keywords"]
            }
        ),
        Tool(
            name="detect_grain",
            description="Detect the granularity of a table (event, session, user, firm, or time level).",
            inputSchema={
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "Name of the table to analyze"
                    },
                    "data_directory": {
                        "type": "string",
                        "description": "Path to data directory"
                    }
                },
                "required": ["table_name"]
            }
        ),
        Tool(
            name="index_knowledge_base",
            description="Index documents in the knowledge base for RAG. Call this after adding new documents.",
            inputSchema={
                "type": "object",
                "properties": {
                    "knowledge_base_path": {
                        "type": "string",
                        "description": "Path to knowledge base directory"
                    }
                }
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """
    Handle tool calls from MCP clients.
    
    Routes each tool call to the appropriate handler.
    """
    logger.info("mcp_tool_call", tool=name, arguments=arguments)
    
    try:
        if name == "discover_feature_context":
            result = await handle_discover_context(arguments)
        elif name == "run_full_pipeline":
            result = await handle_run_pipeline(arguments)
        elif name == "explore_database":
            result = await handle_explore_database(arguments)
        elif name == "find_eligible_tables":
            result = await handle_find_eligible(arguments)
        elif name == "detect_grain":
            result = await handle_detect_grain(arguments)
        elif name == "index_knowledge_base":
            result = await handle_index(arguments)
        else:
            result = {"error": f"Unknown tool: {name}"}
        
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
    
    except Exception as e:
        logger.error("mcp_tool_error", tool=name, error=str(e))
        return [TextContent(
            type="text",
            text=json.dumps({"error": str(e)}, indent=2)
        )]


# --- Tool Handlers ---

async def handle_discover_context(args: dict) -> dict:
    """Handle discover_feature_context tool."""
    from src.agents.context_discovery import ContextDiscoveryAgent
    
    feature_name = args["feature_name"]
    additional_context = args.get("additional_context", "")
    
    agent = ContextDiscoveryAgent()
    result = agent.discover(feature_name, additional_context)
    
    return result.model_dump()


async def handle_run_pipeline(args: dict) -> dict:
    """Handle run_full_pipeline tool."""
    feature_name = args["feature_name"]
    additional_context = args.get("additional_context", "")
    skip_indexing = args.get("skip_indexing", False)
    
    pipeline = ProductSuccessPipeline()
    input_data = PipelineInput(
        feature_name=feature_name,
        additional_context=additional_context,
        skip_indexing=skip_indexing
    )
    
    result = pipeline.run(input_data)
    
    return result.model_dump()


async def handle_explore_database(args: dict) -> dict:
    """Handle explore_database tool."""
    data_directory = args.get("data_directory", "./data/mock_warehouse")
    
    tables = explore_database(data_directory)
    
    return {"tables": [t.model_dump() for t in tables]}


async def handle_find_eligible(args: dict) -> dict:
    """Handle find_eligible_tables tool."""
    keywords = args["keywords"]
    data_directory = args.get("data_directory", "./data/mock_warehouse")
    min_score = args.get("min_score", 0.3)
    
    tables = find_eligible_tables(keywords, data_directory)
    filtered = [t for t in tables if t.score >= min_score]
    
    return {"tables": [t.model_dump() for t in filtered]}


async def handle_detect_grain(args: dict) -> dict:
    """Handle detect_grain tool."""
    table_name = args["table_name"]
    data_directory = args.get("data_directory", "./data/mock_warehouse")
    
    # First explore to get the table metadata
    tables = explore_database(data_directory)
    
    # Find the requested table
    target_table = None
    for table in tables:
        if table.name == table_name:
            target_table = table
            break
    
    if target_table is None:
        return {"error": f"Table '{table_name}' not found"}
    
    result = detect_grain_from_model(target_table)
    
    return result.model_dump()


async def handle_index(args: dict) -> dict:
    """Handle index_knowledge_base tool."""
    knowledge_base_path = args.get("knowledge_base_path", "./knowledge_base")
    
    stats = index_knowledge_base(knowledge_base_path)
    
    return stats


# --- Main Entry Point ---

async def main():
    """
    Run the MCP server.
    
    Uses stdio transport for communication with MCP clients.
    """
    init_observability()
    logger.info("mcp_server_starting")
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
