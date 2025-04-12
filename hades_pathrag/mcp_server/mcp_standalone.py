#!/usr/bin/env python
"""
Standalone MCP (Model Context Protocol) server for HADES-PathRAG.

This is a simplified implementation that follows the Model Context Protocol
standard as defined at https://modelcontextprotocol.io/
"""
import asyncio
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

# Ensure parent directory is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from hades_pathrag.mcp_server.config import get_config
from hades_pathrag.mcp_server.logging_setup import setup_logging
from hades_pathrag.mcp_server.handlers import pathrag_tools


# Initialize logging
logger = logging.getLogger(__name__)


# Tool definitions
TOOLS = [
    {
        "name": "retrieve_path",
        "description": "Retrieve a path from the knowledge graph based on a natural language query",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The natural language query"},
                "start_node": {"type": "string", "description": "Optional starting node"},
                "end_node": {"type": "string", "description": "Optional ending node"},
                "max_length": {"type": "integer", "description": "Maximum path length"},
                "min_similarity": {"type": "number", "description": "Minimum similarity threshold"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "semantic_search",
        "description": "Perform a semantic search on the knowledge graph",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
                "collection": {"type": "string", "description": "Optional specific collection to search"},
                "top_k": {"type": "integer", "description": "Number of results to return"},
                "min_similarity": {"type": "number", "description": "Minimum similarity threshold"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "embed_document",
        "description": "Embed a document and store it in the knowledge graph",
        "inputSchema": {
            "type": "object",
            "properties": {
                "document": {"type": "string", "description": "The document text"},
                "metadata": {"type": "object", "description": "Optional metadata for the document"},
                "model": {"type": "string", "description": "Optional embedding model to use"}
            },
            "required": ["document"]
        }
    },
    {
        "name": "get_graph_statistics",
        "description": "Get statistics about the knowledge graph",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    }
]


# Tool implementations mapping
TOOL_IMPLEMENTATIONS = {
    "retrieve_path": pathrag_tools.retrieve_path,
    "semantic_search": pathrag_tools.semantic_search,
    "embed_document": pathrag_tools.embed_document,
    "get_graph_statistics": pathrag_tools.get_graph_statistics
}


# Create FastAPI app
app = FastAPI(title="HADES-PathRAG MCP Server")


@app.post("/mcp/jsonrpc")
async def handle_jsonrpc(request: Request) -> JSONResponse:
    """
    Handle MCP JSON-RPC requests.
    
    This is the main entry point for MCP clients to interact with the server.
    """
    try:
        body = await request.json()
        
        # Validate request
        if not isinstance(body, dict) or "type" not in body:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid request format"}
            )
        
        # Handle request based on type
        if body["type"] == "request":
            return await handle_mcp_request(body)
        elif body["type"] == "notification":
            await handle_mcp_notification(body)
            return JSONResponse(status_code=204, content=None)
        else:
            return JSONResponse(
                status_code=400,
                content={"error": f"Unsupported message type: {body['type']}"}
            )
    
    except Exception as e:
        logger.exception(f"Error handling JSON-RPC request: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Internal server error: {str(e)}"}
        )


async def handle_mcp_request(request: Dict[str, Any]) -> JSONResponse:
    """
    Handle an MCP request message.
    
    Args:
        request: The MCP request message
        
    Returns:
        A JSON response
    """
    method = request.get("method")
    params = request.get("params", {})
    request_id = request.get("id")
    
    if method == "tools/list":
        # Return the list of available tools
        result = {"tools": TOOLS}
        return JSONResponse(
            status_code=200,
            content={
                "type": "response",
                "id": request_id,
                "result": result
            }
        )
    
    elif method == "tools/call":
        # Call a tool
        tool_name = params.get("name")
        args = params.get("arguments", {})
        
        if not tool_name:
            return JSONResponse(
                status_code=400,
                content={
                    "type": "response",
                    "id": request_id,
                    "error": {
                        "code": -32602,
                        "message": "Tool name not provided"
                    }
                }
            )
        
        tool_func = TOOL_IMPLEMENTATIONS.get(tool_name)
        if not tool_func:
            return JSONResponse(
                status_code=404,
                content={
                    "type": "response",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Tool not found: {tool_name}"
                    }
                }
            )
        
        try:
            # Call the tool function
            result = await tool_func(**args)
            
            # Format the response according to MCP
            content = format_tool_result(result)
            
            return JSONResponse(
                status_code=200,
                content={
                    "type": "response",
                    "id": request_id,
                    "result": {
                        "content": content
                    }
                }
            )
        except Exception as e:
            logger.exception(f"Error calling tool {tool_name}: {e}")
            return JSONResponse(
                status_code=500,
                content={
                    "type": "response",
                    "id": request_id,
                    "error": {
                        "code": -32603,
                        "message": f"Error calling tool: {str(e)}"
                    }
                }
            )
    
    else:
        # Unknown method
        return JSONResponse(
            status_code=404,
            content={
                "type": "response",
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}"
                }
            }
        )


async def handle_mcp_notification(notification: Dict[str, Any]) -> None:
    """
    Handle an MCP notification message.
    
    Args:
        notification: The MCP notification message
    """
    method = notification.get("method")
    params = notification.get("params", {})
    
    # Log the notification
    logger.info(f"Received notification: {method}")


def format_tool_result(result: Any) -> List[Dict[str, Any]]:
    """
    Format a tool result into the expected MCP content format.
    
    Args:
        result: The tool result
        
    Returns:
        A list of content items
    """
    # Handle dictionaries and list results
    if isinstance(result, dict) or isinstance(result, list):
        return [{"type": "text", "text": json.dumps(result, indent=2)}]
    
    # Handle simple types
    if isinstance(result, (str, int, float, bool)):
        return [{"type": "text", "text": str(result)}]
    
    # Fall back to string representation
    return [{"type": "text", "text": str(result)}]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="HADES-PathRAG MCP Server")
    
    parser.add_argument(
        "--config", 
        type=str,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--host", 
        type=str,
        help="Host address to bind to"
    )
    
    parser.add_argument(
        "--port", 
        type=int,
        help="Port to bind to"
    )
    
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Run in debug mode"
    )
    
    parser.add_argument(
        "--log-level", 
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load configuration
    if args.config:
        config = get_config().from_file(args.config)
    else:
        config = get_config()
    
    # Override config with command line arguments
    if args.host:
        config.server.host = args.host
    
    if args.port:
        config.server.port = args.port
    
    if args.debug:
        config.server.debug = args.debug
    
    if args.log_level:
        config.server.log_level = args.log_level
    
    # Setup logging
    setup_logging(config.server.log_level)
    
    logger.info(f"Starting HADES-PathRAG MCP Server (Model Context Protocol)")
    logger.info(f"Server running at http://{config.server.host}:{config.server.port}")
    
    # Run the server with uvicorn
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        log_level=config.server.log_level.lower()
    )


if __name__ == "__main__":
    main()
