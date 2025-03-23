import asyncio
import websockets
import json
import logging
import os
import sys
import signal
from typing import Any, Dict, List, Optional, Callable, TextIO
from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import configuration
from ..config.llm_config import get_llm_config, get_ollama_config, get_model_func_for_provider
from ..config.mcp_config import get_mcp_config, get_arango_config, get_pathrag_config

from ..db.arango_connection import get_db_connection
from ..pathrag.PathRAG import PathRAG
from ..xnx.xnx_pathrag import XnXPathRAG
from ..xnx.xnx_params import XnXParams
from .xnx_tools import (
    mcp0_pathrag_retrieve,
    mcp0_ingest_data,
    mcp0_xnx_pathrag_retrieve,
    mcp0_self_analyze
)

logger = logging.getLogger(__name__)

class MCPTool:
    def __init__(self, name: str, handler: Callable):
        self.name = name
        self.handler = handler

class MCPServer:
    """
    Model Context Protocol (MCP) Server with XnX-enhanced PathRAG integration.
    
    This server implementation combines standard MCP functionality with
    the XnX-enhanced PathRAG system, enabling the "HADES builds HADES"
    recursive pattern.
    """
    
    def __init__(self, host: str = None, port: int = None, 
                 fastapi_port: int = None):
        # Load environment variables
        load_dotenv()
        
        # Load configurations
        self.mcp_config = get_mcp_config()
        self.llm_config = get_llm_config()
        self.ollama_config = get_ollama_config()
        self.arango_config = get_arango_config()
        self.pathrag_config = get_pathrag_config()
        
        # Server config
        server_config = self.mcp_config["server"]
        self.host = host or server_config["host"]
        self.port = port or server_config["port"]
        self.fastapi_port = fastapi_port or server_config["fastapi_port"]
        
        # Initialize PathRAG systems
        self.standard_pathrag = None
        self.xnx_pathrag = None
        
        # Initialize FastAPI for REST endpoint
        self.app = FastAPI(title="HADES-PathRAG MCP Server")
        self.configure_fastapi()
        
        # Registered tools
        self.tools = {}
        self.register_default_tools()
        
        # Session data
        self.sessions = {}
        
        # Signal handlers for graceful shutdown
        self.setup_signal_handlers()
        
    def configure_fastapi(self):
        """Configure FastAPI application with CORS and routes."""
        # Configure CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Register routes
        @self.app.post("/mcp/tools")
        async def handle_tool_call(request: Request):
            """Handle MCP tool call via HTTP."""
            try:
                data = await request.json()
                tool_name = data.get("name")
                parameters = data.get("parameters", {})
                
                if tool_name not in self.tools:
                    raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
                
                # Create session data for this request
                session_id = f"http-{id(request)}"
                session_data = self.sessions.get(session_id, {})
                
                result = await self.tools[tool_name](parameters, session_data)
                return {"result": result}
            except Exception as e:
                logger.error(f"Error processing HTTP request: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/mcp/status")
        async def get_status():
            """Get MCP server status."""
            return {
                "status": "online",
                "tools": list(self.tools.keys()),
                "version": "1.0.0",
                "xnx_enabled": self.xnx_pathrag is not None
            }
        
        @self.app.websocket("/mcp/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for MCP communication."""
            await websocket.accept()
            
            # Create session for this connection
            session_id = f"ws-{id(websocket)}"
            self.sessions[session_id] = {}
            
            try:
                async for message in websocket.iter_text():
                    try:
                        data = json.loads(message)
                        response = await self.process_message(
                            websocket, data, self.sessions[session_id]
                        )
                        await websocket.send_text(json.dumps(response))
                    except json.JSONDecodeError:
                        await websocket.send_text(
                            json.dumps({"error": "Invalid JSON"})
                        )
                    except Exception as e:
                        logger.error(f"Error processing WebSocket message: {str(e)}")
                        await websocket.send_text(
                            json.dumps({"error": str(e)})
                        )
            except WebSocketDisconnect:
                # Clean up session
                if session_id in self.sessions:
                    del self.sessions[session_id]
    
    def register_default_tools(self):
        """Register default MCP tools."""
        self.register_tool("mcp0_pathrag_retrieve", self.pathrag_retrieve)
        self.register_tool("mcp0_ingest_data", self.ingest_data)
        self.register_tool("mcp0_xnx_pathrag_retrieve", self.xnx_pathrag_retrieve)
        self.register_tool("mcp0_self_analyze", self.self_analyze)
    
    def register_tool(self, name: str, handler: Callable):
        """Register an MCP tool."""
        self.tools[name] = handler
        logger.info(f"Registered tool: {name}")
    
    def setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown."""
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))
    
    async def initialize_pathrag(self):
        """Initialize PathRAG systems with Ollama as the default LLM provider."""
        # Import needed functions based on provider configuration
        from ..pathrag.llm import ollama_model_complete, ollama_embed
        
        # Get the Ollama configuration
        ollama_config = self.ollama_config
        provider = self.llm_config["provider"]
        logger.info(f"Initializing PathRAG with LLM provider: {provider}")
        
        # Initialize standard PathRAG
        if not self.standard_pathrag:
            pathrag_config = self.pathrag_config
            self.standard_pathrag = PathRAG(
                working_dir=pathrag_config["cache_dir"],
                embedding_cache_config={"enabled": pathrag_config["embedding_cache"]},
                kv_storage=pathrag_config["kv_storage"],
                vector_storage=pathrag_config["vector_storage"],
                graph_storage=pathrag_config["graph_storage"]
            )
            
            # Configure with Ollama for text generation
            if provider == "ollama":
                logger.info(f"Using Ollama model: {ollama_config['model']}")
                # Set up the model completion function
                self.standard_pathrag.llm_model_func = lambda prompt, **kwargs: ollama_model_complete(
                    prompt=prompt,
                    hashing_kv={"global_config": {"llm_model_name": ollama_config['model']}},
                    host=ollama_config['host'],
                    timeout=ollama_config['timeout'],
                    **ollama_config['parameters']
                )
                
                # Set up the embedding function
                self.standard_pathrag.set_embedding_function(
                    lambda texts, **kwargs: ollama_embed(
                        texts=texts,
                        embed_model=ollama_config['embed_model'],
                        host=ollama_config['host'],
                        timeout=ollama_config['timeout']
                    )
                )
        
        # Initialize XnX-enhanced PathRAG
        if not self.xnx_pathrag:
            arango_config = self.arango_config
            self.xnx_pathrag = XnXPathRAG(
                db_url=arango_config["url"],
                db_name=arango_config["db"],
                username=arango_config["user"],
                password=arango_config["password"]
            )
            
            # Configure with Ollama for text generation
            if provider == "ollama":
                # Set the XnX parameters to use Ollama
                xnx_params = XnXParams(
                    llm_provider="ollama",
                    llm_model=ollama_config['model'],
                    embed_model=ollama_config['embed_model']
                )
                self.xnx_pathrag.set_parameters(xnx_params)
    
    async def process_message(self, websocket, data: Dict[str, Any], session_data: Dict[str, Any]):
        """
        Process a message from a client.
        
        Args:
            websocket: WebSocket connection
            data: Message data
            session_data: Session-specific data
            
        Returns:
            Response to be sent back to the client
        """
        method = data.get("method")
        params = data.get("params", {})
        id = data.get("id")
        
        response = {
            "jsonrpc": "2.0",
            "id": id
        }
        
        try:
            if method == "initialize":
                # Get the protocol version from the client request
                protocol_version = params.get("protocolVersion", "2024-11-05")
                logger.info(f"Client requested protocol version: {protocol_version}")
                
                # Check if server support that version
                if protocol_version != "2024-11-05":
                    logger.warning(f"Unsupported protocol version: {protocol_version}")
                
                # Return capabilities
                response["result"] = {
                    "capabilities": {
                        "toolCalls": self.describe_tools()
                    }
                }
            elif method == "toolCall":
                tool_id = params.get("toolId")
                tool_name = params.get("toolName")
                tool_params = params.get("parameters", {})
                
                if tool_name not in self.tools:
                    raise ValueError(f"Unknown tool: {tool_name}")
                
                # Execute the tool
                result = await self.tools[tool_name](tool_params, session_data)
                response["result"] = {
                    "toolCallId": tool_id,
                    "result": result
                }
            else:
                raise ValueError(f"Unknown method: {method}")
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            response["error"] = {
                "code": -32603,
                "message": f"Internal error: {str(e)}"
            }
        
        return response
    
    async def pathrag_retrieve(self, params: Dict[str, Any], session_data: Dict[str, Any]):
        """
        Tool handler to retrieve paths using standard PathRAG.
        
        Args:
            params: Tool parameters
            session_data: Session-specific data
            
        Returns:
            Retrieved paths
        """
        await self.initialize_pathrag()
        
        query = params.get("query")
        domain_filter = params.get("domain_filter")
        as_of_version = params.get("as_of_version")
        
        if not query:
            raise ValueError("Missing required parameter: query")
        
        # Initialize query parameters
        query_param = {}
        
        # Call standard PathRAG
        results = await self.standard_pathrag.aquery(query, query_param)
        
        # Filter by domain if specified
        if domain_filter:
            results = [r for r in results if domain_filter in r.get("domains", [])]
        
        return {
            "paths": results,
            "query": query,
            "domain_filter": domain_filter,
            "as_of_version": as_of_version
        }
    
    async def xnx_pathrag_retrieve(self, params: Dict[str, Any], session_data: Dict[str, Any]):
        """
        Tool handler to retrieve paths using XnX-enhanced PathRAG.
        
        Args:
            params: Tool parameters
            session_data: Session-specific data
            
        Returns:
            Retrieved paths with XnX parameters
        """
        await self.initialize_pathrag()
        
        query = params.get("query")
        min_weight = float(params.get("min_weight", 0.5))
        max_distance = int(params.get("max_distance", 3))
        direction = params.get("direction")
        valid_from = params.get("valid_from")
        valid_until = params.get("valid_until")
        domain_filter = params.get("domain_filter")
        
        if not query:
            raise ValueError("Missing required parameter: query")
        
        # Create XnX parameters
        xnx_params = XnXParams(
            min_weight=min_weight,
            max_distance=max_distance,
            direction=direction,
            valid_from=valid_from,
            valid_until=valid_until
        )
        
        # Query using XnX-enhanced PathRAG
        paths = await self.xnx_pathrag.query(query, xnx_params)
        
        # Filter by domain if specified
        if domain_filter:
            paths = [p for p in paths if domain_filter in p.get("domains", [])]
        
        return {
            "paths": paths,
            "query_parameters": {
                "min_weight": min_weight,
                "max_distance": max_distance,
                "direction": direction,
                "valid_from": valid_from,
                "valid_until": valid_until,
                "domain_filter": domain_filter
            }
        }
    
    async def ingest_data(self, params: Dict[str, Any], session_data: Dict[str, Any]):
        """
        Tool handler to ingest data into the knowledge graph.
        
        Args:
            params: Tool parameters
            session_data: Session-specific data
            
        Returns:
            Ingestion results
        """
        await self.initialize_pathrag()
        
        data = params.get("data")
        domain = params.get("domain", "general")
        as_of_version = params.get("as_of_version")
        
        if not data:
            raise ValueError("Missing required parameter: data")
        
        if not isinstance(data, list):
            data = [data]
        
        # Process each data point
        results = []
        for item in data:
            # Add domain if not present
            if "domains" not in item:
                item["domains"] = [domain]
            elif domain not in item["domains"]:
                item["domains"].append(domain)
            
            # Add version if specified
            if as_of_version:
                item["version"] = as_of_version
            
            # Ingest with standard PathRAG
            standard_result = await self.standard_pathrag.ainsert(item)
            
            # Ingest with XnX-enhanced PathRAG if available
            xnx_result = None
            if self.xnx_pathrag:
                xnx_result = await self.xnx_pathrag.ingest_data(item)
            
            results.append({
                "item": item,
                "standard_result": standard_result,
                "xnx_result": xnx_result
            })
        
        return {
            "results": results,
            "count": len(results),
            "domain": domain,
            "as_of_version": as_of_version
        }
    
    async def self_analyze(self, params: Dict[str, Any], session_data: Dict[str, Any]):
        """
        Tool handler for HADES to analyze its own codebase.
        
        Args:
            params: Tool parameters
            session_data: Session-specific data
            
        Returns:
            Self-analysis results
        """
        await self.initialize_pathrag()
        
        query = params.get("query")
        target_components = params.get("target_components")
        min_confidence = float(params.get("min_confidence", 0.7))
        
        if not query:
            raise ValueError("Missing required parameter: query")
        
        # Use XnX-enhanced PathRAG for self-analysis
        if not self.xnx_pathrag:
            raise ValueError("XnX-enhanced PathRAG is required for self-analysis but is not initialized")
        
        # Build self-referential query
        if target_components:
            query = f"{query} Focus on the following components: {', '.join(target_components)}"
        
        # Execute query with XnX parameters
        paths = await self.xnx_pathrag.query(
            query,
            XnXParams(
                min_weight=min_confidence,
                max_distance=3,
                direction=-1  # Only consider outbound relationships for code analysis
            )
        )
        
        # Analyze code quality (placeholder for now)
        code_quality = {
            "complexity": 0.85,
            "maintainability": 0.78,
            "test_coverage": 0.65,
            "performance": 0.92
        }
        
        # Generate improvement suggestions (placeholder for now)
        suggestions = [
            {
                "metric": "test_coverage",
                "current_value": 0.65,
                "target_value": 0.8,
                "suggestion": "Add tests for the PathRAG query methods",
                "affected_files": ["src/pathrag/PathRAG.py"]
            }
        ]
        
        return {
            "paths": paths,
            "code_quality": code_quality,
            "improvement_suggestions": suggestions,
            "query_parameters": {
                "query": query,
                "target_components": target_components,
                "min_confidence": min_confidence
            }
        }
    
    def describe_tools(self):
        """Generate tool descriptions in MCP format."""
        tool_descriptions = []
        
        # Describe the ingest_data tool
        tool_descriptions.append({
            "name": "mcp0_ingest_data",
            "description": "Ingest data into the knowledge graph using PathRAG",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {
                        "description": "List of data points to ingest",
                        "type": "array"
                    },
                    "domain": {
                        "description": "Domain to associate with the data",
                        "type": "string",
                        "default": "general"
                    },
                    "as_of_version": {
                        "description": "Optional version to tag the data with",
                        "type": "string"
                    }
                },
                "required": ["data"]
            }
        })
        
        # Describe the pathrag_retrieve tool
        tool_descriptions.append({
            "name": "mcp0_pathrag_retrieve",
            "description": "Retrieve paths from the knowledge graph using PathRAG",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "description": "The query to retrieve paths for",
                        "type": "string"
                    },
                    "domain_filter": {
                        "description": "Optional domain filter",
                        "type": "string"
                    },
                    "as_of_version": {
                        "description": "Optional version to query against",
                        "type": "string"
                    }
                },
                "required": ["query"]
            }
        })
        
        # Describe the xnx_pathrag_retrieve tool
        tool_descriptions.append({
            "name": "mcp0_xnx_pathrag_retrieve",
            "description": "Retrieve paths from the knowledge graph using XnX-enhanced PathRAG",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "description": "The query to retrieve paths for",
                        "type": "string"
                    },
                    "min_weight": {
                        "description": "Minimum relationship weight (0.0-1.0)",
                        "type": "number",
                        "default": 0.5
                    },
                    "max_distance": {
                        "description": "Maximum number of hops",
                        "type": "integer",
                        "default": 3
                    },
                    "direction": {
                        "description": "Path direction (positive=inbound, negative=outbound)",
                        "type": "integer"
                    },
                    "valid_from": {
                        "description": "Start of validity period (ISO date)",
                        "type": "string"
                    },
                    "valid_until": {
                        "description": "End of validity period (ISO date)",
                        "type": "string"
                    },
                    "domain_filter": {
                        "description": "Optional domain filter",
                        "type": "string"
                    }
                },
                "required": ["query"]
            }
        })
        
        # Describe the self_analyze tool
        tool_descriptions.append({
            "name": "mcp0_self_analyze",
            "description": "Analyze HADES codebase to suggest improvements",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "description": "The analysis query",
                        "type": "string"
                    },
                    "target_components": {
                        "description": "List of components to analyze",
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    },
                    "min_confidence": {
                        "description": "Minimum confidence threshold",
                        "type": "number",
                        "default": 0.7
                    }
                },
                "required": ["query"]
            }
        })
        
        return tool_descriptions
    
    async def run_socket(self):
        """Run the MCP server with WebSocket transport."""
        logger.info(f"Starting MCP server on {self.host}:{self.port}")
        await self.initialize_pathrag()
        
        async with websockets.serve(self.handle_websocket, self.host, self.port):
            await asyncio.Future()  # Run forever
    
    async def handle_websocket(self, websocket, path):
        """Handle a WebSocket connection."""
        # Create session for this connection
        session_id = f"ws-{id(websocket)}"
        self.sessions[session_id] = {}
        
        try:
            async for message in websocket.iter_text():
                try:
                    data = json.loads(message)
                    response = await self.process_message(
                        websocket, data, self.sessions[session_id]
                    )
                    await websocket.send(json.dumps(response))
                except json.JSONDecodeError:
                    await websocket.send(
                        json.dumps({"error": "Invalid JSON"})
                    )
                except Exception as e:
                    logger.error(f"Error processing WebSocket message: {str(e)}")
                    await websocket.send(
                        json.dumps({"error": str(e)})
                    )
        except websockets.exceptions.ConnectionClosed:
            # Clean up session
            if session_id in self.sessions:
                del self.sessions[session_id]
    
    async def run_stdio(self):
        """Run the MCP server using stdio transport for Windsurf integration."""
        logger.info("Starting MCP server with stdio transport")
        logger.info(f"Registered tools: {', '.join(self.tools.keys())}")
        
        # Initialize PathRAG at startup
        await self.initialize_pathrag()
        
        # Set up stdio
        stdin = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(stdin)
        await asyncio.get_event_loop().connect_read_pipe(
            lambda: protocol, sys.stdin
        )
        
        stdout = sys.stdout
        
        # Clean the output for stdio MCP protocol
        sys.stdout = open(os.devnull, "w")
        
        # Create a session for the stdio transport
        session_id = "stdio"
        self.sessions[session_id] = {}
        
        # Send initial message indicating we're ready
        header = f"Content-Length: {len('{}')}\r\n\r\n"
        stdout.write(header + "{}")
        stdout.flush()
        
        # Process messages
        while True:
            # Read headers
            header_line = await stdin.readline()
            if not header_line:
                break
                
            header_line = header_line.decode("utf-8").strip()
            if not header_line:
                continue
                
            # Get content length
            content_length = 0
            if header_line.startswith("Content-Length: "):
                content_length = int(header_line.split(": ")[1])
                
            # Skip additional headers
            while True:
                header_line = await stdin.readline()
                if not header_line:
                    break
                    
                header_line = header_line.decode("utf-8").strip()
                if not header_line:
                    break
            
            # Read content
            if content_length > 0:
                content = await stdin.readexactly(content_length)
                content = content.decode("utf-8")
                
                try:
                    data = json.loads(content)
                    response = await self.process_stdio_message(
                        data, self.sessions[session_id]
                    )
                    
                    response_str = json.dumps(response)
                    header = f"Content-Length: {len(response_str)}\r\n\r\n"
                    stdout.write(header + response_str)
                    stdout.flush()
                except json.JSONDecodeError:
                    error_response = json.dumps({"error": "Invalid JSON"})
                    header = f"Content-Length: {len(error_response)}\r\n\r\n"
                    stdout.write(header + error_response)
                    stdout.flush()
                except Exception as e:
                    logger.error(f"Error processing stdio message: {str(e)}")
                    error_response = json.dumps({"error": str(e)})
                    header = f"Content-Length: {len(error_response)}\r\n\r\n"
                    stdout.write(header + error_response)
                    stdout.flush()
    
    async def process_stdio_message(self, data: Dict[str, Any], session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a JSON-RPC message from stdio and return a response."""
        method = data.get("method")
        params = data.get("params", {})
        id = data.get("id")
        
        response = {
            "jsonrpc": "2.0",
            "id": id
        }
        
        try:
            if method == "initialize":
                # Get the protocol version from the client request
                protocol_version = params.get("protocolVersion", "2024-11-05")
                logger.info(f"Client requested protocol version: {protocol_version}")
                
                # Check if server support that version
                if protocol_version != "2024-11-05":
                    logger.warning(f"Unsupported protocol version: {protocol_version}")
                
                # Return capabilities
                response["result"] = {
                    "capabilities": {
                        "toolCalls": self.describe_tools()
                    }
                }
            elif method == "toolCall":
                tool_id = params.get("toolId")
                tool_name = params.get("toolName")
                tool_params = params.get("parameters", {})
                
                if tool_name not in self.tools:
                    raise ValueError(f"Unknown tool: {tool_name}")
                
                # Execute the tool
                result = await self.tools[tool_name](tool_params, session_data)
                response["result"] = {
                    "toolCallId": tool_id,
                    "result": result
                }
            elif method == "shutdown":
                # Shutdown the server
                response["result"] = None
                asyncio.create_task(self.shutdown())
            else:
                raise ValueError(f"Unknown method: {method}")
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            response["error"] = {
                "code": -32603,
                "message": f"Internal error: {str(e)}"
            }
        
        return response
    
    async def run_fastapi(self):
        """Run the MCP server with FastAPI for HTTP interface."""
        logger.info(f"Starting FastAPI server on {self.host}:{self.fastapi_port}")
        await self.initialize_pathrag()
        
        import uvicorn
        config = uvicorn.Config(self.app, host=self.host, port=self.fastapi_port)
        server = uvicorn.Server(config)
        await server.serve()
    
    async def run(self, transport_type="socket"):
        """
        Run the MCP server with the specified transport.
        
        Args:
            transport_type: Transport type (socket, stdio, fastapi)
        """
        if transport_type == "socket":
            await self.run_socket()
        elif transport_type == "stdio":
            await self.run_stdio()
        elif transport_type == "fastapi":
            await self.run_fastapi()
        else:
            raise ValueError(f"Unknown transport type: {transport_type}")
    
    async def shutdown(self):
        """Gracefully shut down the server."""
        logger.info("Shutting down MCP server")
        # Perform any cleanup here
        asyncio.get_running_loop().stop()


# Server instance to be used by other modules
mcp_server = MCPServer()

def main():
    """Entry point for Poetry script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the MCP server")
    parser.add_argument(
        "--transport", 
        choices=["socket", "stdio", "fastapi"], 
        default="socket",
        help="Transport type to use"
    )
    parser.add_argument(
        "--host", 
        default="localhost",
        help="Host to bind to"
    )
    parser.add_argument(
        "--port", 
        type=int,
        default=8765,
        help="Port to bind to for WebSocket transport"
    )
    parser.add_argument(
        "--fastapi-port", 
        type=int,
        default=8123,
        help="Port to bind to for FastAPI transport"
    )
    
    args = parser.parse_args()
    
    # Configure server
    mcp_server.host = args.host
    mcp_server.port = args.port
    mcp_server.fastapi_port = args.fastapi_port
    
    # Run server
    asyncio.run(mcp_server.run(args.transport))


# Entry point when running this file directly
if __name__ == "__main__":
    main()
