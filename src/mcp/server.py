"""
MCP server implementation for HADES-PathRAG with Ollama integration.

This module provides a FastAPI server that exposes MCP tools for XnX-enhanced
PathRAG, with Ollama integration for local LLM inference.
"""

import os
import sys
import json
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import httpx
from fastapi import FastAPI, HTTPException, Depends, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field
import uvicorn

# Try to import from our structured location
try:
    from src.xnx import XnXPathRAG, XnXQueryParams, XnXIdentityToken
    from src.mcp import xnx_tools  # Import our XnX tools
except ImportError:
    # Fall back to old_hades_imports
    sys.path.append('old_hades_imports')
    sys.path.append('.')
    from src.xnx import XnXPathRAG, XnXQueryParams, XnXIdentityToken
    from src.mcp import xnx_tools

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="HADES-PathRAG MCP Server")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, limit this to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
current_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(current_dir, "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Ollama integration
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")

# Pydantic models for API
class OllamaCompletionRequest(BaseModel):
    prompt: str
    model: str = Field(default=OLLAMA_MODEL)
    system: Optional[str] = None
    template: Optional[str] = None
    context: Optional[List[int]] = None
    options: Optional[Dict[str, Any]] = None
    format: Optional[str] = None

class XnXPathQuery(BaseModel):
    query: str
    domain_filter: Optional[str] = None
    min_weight: float = 0.5
    max_distance: int = 3
    direction: Optional[int] = None
    as_of_version: Optional[str] = None

class XnXRelationship(BaseModel):
    from_entity: str
    to_entity: str
    weight: float = 1.0
    direction: int = -1
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class XnXIdentityRequest(BaseModel):
    user_id: str
    object_id: str
    duration_minutes: int = 60

class XnXAccessCheck(BaseModel):
    user_id: str
    resource_id: str
    min_weight: float = 0.7
    identity_token_id: Optional[str] = None

# Global XnXPathRAG instance
_xnx_pathrag = None

def get_xnx_pathrag():
    """Get or create XnXPathRAG instance."""
    global _xnx_pathrag
    if _xnx_pathrag is None:
        # Initialize with Ollama LLM function
        _xnx_pathrag = XnXPathRAG(
            working_dir="./path_cache",
            llm_model_func=ollama_generate,
            arango_adapter=None  # Will create default
        )
    return _xnx_pathrag

async def ollama_generate(prompt: str, system_prompt: Optional[str] = None) -> str:
    """Generate text using Ollama LLM.
    
    Args:
        prompt: The prompt to send to Ollama
        system_prompt: Optional system prompt for context
        
    Returns:
        Generated text response
    """
    async with httpx.AsyncClient() as client:
        request_data = {
            "model": OLLAMA_MODEL,
            "prompt": prompt
        }
        
        if system_prompt:
            request_data["system"] = system_prompt
            
        try:
            response = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json=request_data,
                timeout=60.0
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except Exception as e:
            logger.error(f"Error calling Ollama: {str(e)}")
            return f"Error generating response: {str(e)}"

# MCP endpoints
@app.post("/mcp/ollama/generate")
async def mcp_ollama_generate(request: OllamaCompletionRequest):
    """Generate text using Ollama LLM."""
    try:
        response = await ollama_generate(request.prompt, request.system)
        return {"response": response}
    except Exception as e:
        logger.error(f"Error in Ollama generation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating response: {str(e)}"
        )

@app.post("/mcp/pathrag/retrieve")
async def mcp_pathrag_retrieve(query: XnXPathQuery):
    """Retrieve paths from the knowledge graph using XnX PathRAG."""
    try:
        result = xnx_tools.mcp0_xnx_pathrag_retrieve(
            query=query.query,
            domain_filter=query.domain_filter,
            min_weight=query.min_weight,
            max_distance=query.max_distance,
            direction=query.direction,
            as_of_version=query.as_of_version
        )
        return result
    except Exception as e:
        logger.error(f"Error in PathRAG retrieval: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving paths: {str(e)}"
        )

@app.post("/mcp/pathrag/create_relationship")
async def mcp_create_relationship(relationship: XnXRelationship):
    """Create a relationship with XnX notation between entities."""
    try:
        result = xnx_tools.mcp0_xnx_create_relationship(
            from_entity=relationship.from_entity,
            to_entity=relationship.to_entity,
            weight=relationship.weight,
            direction=relationship.direction,
            valid_from=relationship.valid_from,
            valid_until=relationship.valid_until,
            metadata=relationship.metadata
        )
        return result
    except Exception as e:
        logger.error(f"Error creating relationship: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error creating relationship: {str(e)}"
        )

@app.post("/mcp/pathrag/assume_identity")
async def mcp_assume_identity(request: XnXIdentityRequest):
    """Create an identity assumption token for a user to act as an object."""
    try:
        result = xnx_tools.mcp0_xnx_assume_identity(
            user_id=request.user_id,
            object_id=request.object_id,
            duration_minutes=request.duration_minutes
        )
        return result
    except Exception as e:
        logger.error(f"Error assuming identity: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error assuming identity: {str(e)}"
        )

@app.post("/mcp/pathrag/verify_access")
async def mcp_verify_access(request: XnXAccessCheck):
    """Verify if a user has access to a resource using XnX access control."""
    try:
        result = xnx_tools.mcp0_xnx_verify_access(
            user_id=request.user_id,
            resource_id=request.resource_id,
            min_weight=request.min_weight,
            identity_token_id=request.identity_token_id
        )
        return result
    except Exception as e:
        logger.error(f"Error verifying access: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error verifying access: {str(e)}"
        )

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend HTML."""
    html_file = os.path.join(static_dir, "index.html")
    return FileResponse(html_file)

@app.get("/api")
async def api_info():
    """API information endpoint."""
    return {
        "name": "HADES-PathRAG MCP Server",
        "version": "1.0.0",
        "description": "XnX-enhanced PathRAG for HADES",
        "ollama_model": OLLAMA_MODEL,
        "endpoints": [
            "/mcp/ollama/generate",
            "/mcp/pathrag/retrieve",
            "/mcp/pathrag/create_relationship",
            "/mcp/pathrag/assume_identity",
            "/mcp/pathrag/verify_access"
        ]
    }

def start_server(host="0.0.0.0", port=8000):
    """Start the MCP server."""
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_server()
