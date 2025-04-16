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
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Awaitable, TypeVar, cast, Union

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


# Define tool function type
ToolFunc = Callable[..., Awaitable[Any]]

# Tool implementations mapping
TOOL_IMPLEMENTATIONS: Dict[str, ToolFunc] = {
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
        # Set a log marker for better tracing
        request_id = f"req_{int(time.time())}_{id(request)}"
        logger.debug(f"🔄 [{request_id}] Received new JSON-RPC request")
        
        # Parse the request body with error handling
        try:
            body = await request.json()
            logger.debug(f"📄 [{request_id}] Request body: {body}")
        except Exception as json_err:
            logger.error(f"❌ [{request_id}] Failed to parse JSON: {json_err}")
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid JSON in request body"}
            )
        
        # Validate request format
        if not isinstance(body, dict):
            logger.error(f"❌ [{request_id}] Request body is not a dictionary")
            return JSONResponse(
                status_code=400,
                content={"error": "Request body must be a JSON object"}
            )
        
        if "type" not in body:
            logger.error(f"❌ [{request_id}] Request missing 'type' field")
            return JSONResponse(
                status_code=400,
                content={"error": "Missing 'type' field in request"}
            )
        
        # Handle request based on type - with specific handling for tools/list
        if body["type"] == "request" and body.get("method") == "tools/list":
            # Fast path for tool listing - most frequent and simplest operation
            logger.debug(f"🔍 [{request_id}] Processing tools/list request")
            req_id = body.get("id", "unknown")
            
            # Return tools list directly without further processing
            result = {"tools": TOOLS}
            response_content = {
                "type": "response",
                "id": req_id,
                "result": result
            }
            logger.debug(f"✅ [{request_id}] Completed tools/list request")
            return JSONResponse(status_code=200, content=response_content)
        elif body["type"] == "request":
            # Process other requests through the normal handler
            logger.debug(f"🔄 [{request_id}] Processing regular request: {body.get('method')}")
            try:
                # Get the response from the handler
                from typing import Dict, Any, Union
                from fastapi.responses import JSONResponse
                request_result: Union[Dict[str, Any], JSONResponse] = await handle_mcp_request(body)
                logger.debug(f"✅ [{request_id}] Completed request")
                
                # Ensure we're returning a JSONResponse (to satisfy type checker)
                if isinstance(request_result, JSONResponse):
                    # It's already a JSONResponse
                    return request_result
                else:
                    # Convert any result (dict or otherwise) to JSONResponse
                    return JSONResponse(status_code=200, content={"result": request_result} if not isinstance(request_result, dict) else request_result)
            except asyncio.TimeoutError:
                logger.error(f"⏱️ [{request_id}] Request timed out")
                return JSONResponse(
                    status_code=504,  # Gateway Timeout
                    content={
                        "type": "response",
                        "id": body.get("id"),
                        "error": {
                            "code": -32001,
                            "message": "Request timed out"
                        }
                    }
                )
        elif body["type"] == "notification":
            # Process notifications
            logger.debug(f"📢 [{request_id}] Processing notification")
            await handle_mcp_notification(body)
            logger.debug(f"✅ [{request_id}] Completed notification")
            return JSONResponse(status_code=204, content=None)
        else:
            # Handle unsupported message types
            logger.error(f"❌ [{request_id}] Unsupported message type: {body['type']}")
            return JSONResponse(
                status_code=400,
                content={
                    "error": f"Unsupported message type: {body['type']}"
                }
            )
    except Exception as e:
        # Log the full exception with traceback
        import traceback
        logger.error(f"❌ Error handling JSON-RPC request: {e}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Internal server error: {str(e)}"
            }
        )


async def handle_mcp_request(request: Dict[str, Any]) -> Union[Dict[str, Any], JSONResponse]:
    """
    Handle an MCP request message.
    
    Args:
        request: The MCP request message
        
    Returns:
        A JSON response
    """
    # Generate a unique trace ID for this request for tracking across logs
    trace_id = f"trace_{int(time.time())}_{id(request)}"
    
    method = request.get("method")
    params = request.get("params", {})
    request_id = request.get("id")
    
    logger.debug(f"[{trace_id}] 🔍 Processing request: method={method}, id={request_id}")
    
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
        
        tool_func: Optional[ToolFunc] = TOOL_IMPLEMENTATIONS.get(tool_name)
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
            # Call the tool function with proper type casting and timeout handling
            tool_callable = cast(ToolFunc, tool_func)
            
            # Add timeout handling to prevent server hanging
            config = get_config()
            timeout_seconds = getattr(config.server, "timeout_seconds", 60)  # Default to 60s if not configured
            
            # Log the request details for debugging
            logger.debug(f"Calling tool '{tool_name}' with args: {args}")
            logger.debug(f"Using timeout of {timeout_seconds} seconds")
            
            # Create a task with timeout
            try:
                # Add more detailed logging for troubleshooting
                start_time = time.time()
                logger.debug(f"⏱️ Starting tool '{tool_name}' at {time.strftime('%H:%M:%S')}")
                
                # Add progress tracking for long-running operations
                progress_task = None
                
                # Create helper function for periodic progress logging
                progress_counter = 0  # Define this outside the function
                
                # Extract the log progress function to avoid nested function issues
                async def _log_tool_progress(tool: str, start: float) -> None:
                    nonlocal progress_counter
                    progress_counter += 1
                    elapsed = time.time() - start
                    logger.debug(f"⏳ Tool '{tool}' still running after {elapsed:.2f}s (progress check #{progress_counter})")
                
                # Create the tracking loop function
                async def log_progress() -> None:
                    while True:
                        await asyncio.sleep(5.0)  # Log every 5 seconds
                        await _log_tool_progress(tool_name, start_time)
                
                # Start progress tracking
                progress_task = asyncio.create_task(log_progress())
                
                # Use asyncio.wait_for to apply a timeout
                try:
                    result = await asyncio.wait_for(
                        tool_callable(**args),
                        timeout=timeout_seconds
                    )
                finally:
                    # Clean up progress task if it's still running
                    if progress_task and not progress_task.done():
                        progress_task.cancel()
                        try:
                            await progress_task
                        except asyncio.CancelledError:
                            pass
                
                elapsed_time = time.time() - start_time
                logger.debug(f"✅ Tool '{tool_name}' completed successfully in {elapsed_time:.2f}s")
            except asyncio.TimeoutError:
                elapsed = time.time() - start_time
                logger.error(f"⏱️ Tool '{tool_name}' TIMED OUT after {elapsed:.2f}s (limit: {timeout_seconds}s)")
                logger.error(f"TIMEOUT DETAILS: tool={tool_name}, args={args}")
                return JSONResponse(
                    status_code=504,  # Gateway Timeout
                    content={
                        "type": "response",
                        "id": request_id,
                        "error": {
                            "code": -32001,
                            "message": f"Request timed out after {timeout_seconds} seconds"
                        }
                    }
                )
            
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
            import traceback
            logger.error(f"❌ Error calling tool {tool_name}: {e}")
            logger.error(f"EXCEPTION DETAILS: tool={tool_name}, args={args}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
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
        A list of content items in the MCP format
    """
    # Handle None values safely
    if result is None:
        return [{"type": "text", "text": ""}]
        
    # Handle dictionaries and list results
    if isinstance(result, dict) or isinstance(result, list):
        try:
            # Use a safer JSON serialization approach
            text = json.dumps(result, indent=2, default=str)
            return [{"type": "text", "text": text}]
        except Exception as e:
            # Fallback for non-serializable objects
            logger.warning(f"Error serializing result to JSON: {e}")
            return [{"type": "text", "text": f"Result (not JSON serializable): {str(result)}"}]
    
    # Handle simple types
    if isinstance(result, (str, int, float, bool)):
        return [{"type": "text", "text": str(result)}]
    
    # Fall back to string representation with type information
    return [{"type": "text", "text": f"Result ({type(result).__name__}): {str(result)}"}]


def parse_args() -> argparse.Namespace:
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


def main() -> None:
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
