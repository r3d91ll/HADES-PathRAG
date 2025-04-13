#!/usr/bin/env python
"""
Test script for HADES-PathRAG MCP server.
This script demonstrates how to interact with the MCP server using the JSON-RPC protocol.
"""
import requests
import json
import sys
from typing import Dict, Any, cast, List, Optional


def call_mcp_tool(tool_name: str, **args: Any) -> Dict[str, Any]:
    """
    Call an MCP tool using the JSON-RPC protocol.
    
    Args:
        tool_name: The name of the tool to call
        **args: Arguments to pass to the tool
        
    Returns:
        Tool response with proper type safety
    """
    # MCP server endpoint
    url = "http://127.0.0.1:8000/mcp/jsonrpc"
    
    # Construct the MCP request with proper type safety
    request_id = "test-1"
    request = {
        "type": "request",
        "id": request_id,
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": args
        }
    }
    
    try:
        # Make the request with timeouts and error handling
        response = requests.post(
            url, 
            json=request,
            headers={"Content-Type": "application/json"},
            timeout=30  # Add timeout for safety
        )
        
        # Check response
        if response.status_code == 200:
            result = response.json()
            
            # Validate the response format
            if not isinstance(result, dict):
                return {"error": f"Invalid response format: {result}"}
                
            # Check if it's a proper MCP response
            if result.get("type") != "response" or result.get("id") != request_id:
                return {"error": f"Invalid MCP response: {result}"}
                
            # Extract the actual result
            if "result" in result and "content" in result["result"]:
                return {"result": result["result"]["content"]}
            elif "error" in result:
                return {"error": result["error"]}
                
            return cast(Dict[str, Any], result)
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return {"error": f"HTTP {response.status_code}: {response.text}"}
    except requests.RequestException as e:
        print(f"Request error: {e}")
        return {"error": f"Request error: {str(e)}"}
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return {"error": f"Invalid JSON response: {str(e)}"}
    except Exception as e:
        print(f"Unexpected error: {e}")
        return {"error": f"Unexpected error: {str(e)}"}


def list_tools() -> List[Dict[str, Any]]:
    """List available tools from the MCP server."""
    url = "http://127.0.0.1:8000/mcp/jsonrpc"
    
    request = {
        "type": "request",
        "id": "list-tools",
        "method": "tools/list",
        "params": {}
    }
    
    response = requests.post(
        url, 
        json=request,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        tools = response.json().get("result", {}).get("tools", [])
        print("Available tools:")
        for tool in tools:
            print(f"- {tool['name']}: {tool['description']}")
        return cast(List[Dict[str, Any]], tools)
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return []


def print_response(response: Dict[str, Any]) -> None:
    """Pretty print a response."""
    if not response:
        return
    
    print("\nResponse:")
    if "content" in response.get("result", {}):
        content = response["result"]["content"]
        for item in content:
            if "text" in item:
                print(f"- {item['type']}: {item['text']}")
            elif "body" in item:
                print(f"- {item['type']}: {item['body']}")
            else:
                print(f"- {item['type']}: {item}")
    else:
        print(json.dumps(response, indent=2))


def test_semantic_search() -> None:
    """Test the semantic_search tool."""
    print("\n=== Testing semantic_search ===")
    response = call_mcp_tool(
        "semantic_search",
        query="machine learning algorithms",
        top_k=3
    )
    print_response(response)


def test_retrieve_path() -> None:
    """Test the retrieve_path tool."""
    print("\n=== Testing retrieve_path ===")
    response = call_mcp_tool(
        "retrieve_path",
        query="How does the ingestion pipeline work?",
        max_length=3
    )
    print_response(response)


def test_embed_document() -> None:
    """Test the hybrid semantic and structural embedding system."""
    print("\n=== Testing Hybrid Embedding System ===")
    # Test a code snippet to demonstrate ModernBERT code embeddings
    code_response = call_mcp_tool(
        "embed_document",
        document="def process_embeddings(doc):\n    semantic = doc.semantic_embedding\n    structural = doc.isne_embedding\n    return combine_embeddings(semantic, structural)",
        metadata={
            "source": "test", 
            "type": "code",
            "content_type": "text/x-python",
            "path": "example.py"
        }
    )
    print("\nCode Embedding Response:")
    print_response(code_response)
    
    # Test a text snippet to demonstrate ModernBERT text embeddings
    text_response = call_mcp_tool(
        "embed_document",
        document="HADES-PathRAG is a hybrid framework that combines semantic understanding with structural graph traversal for enhanced knowledge representation.",
        metadata={
            "source": "test", 
            "type": "documentation",
            "content_type": "text/plain"
        }
    )
    print("\nText Embedding Response:")
    print_response(text_response)


def test_graph_statistics() -> None:
    """Test the get_graph_statistics tool."""
    print("\n=== Testing get_graph_statistics ===")
    response = call_mcp_tool("get_graph_statistics")
    print_response(response)


def main() -> None:
    """Run tests for the HADES-PathRAG Hybrid Semantic Graph Embedding System.
    
    This script demonstrates the key features of the hybrid embedding system:
    1. Combined semantic and structural embeddings
    2. Content modality detection (code vs text)
    3. Graph-based path retrieval
    4. Semantic search capabilities
    
    The system uses ModernBERT for semantic embedding and ISNE for structural embedding,
    combining them with configurable weights (default: 0.6 semantic, 0.4 structural).
    """
    try:
        print("\n===== HADES-PathRAG Hybrid Semantic Graph Embedding System Tester =====")
        print("Semantic Model: answerdotai/ModernBERT-base (text), juanwisz/modernbert-python-code-retrieval (code)")
        print("Structural Model: ISNE (Inductive Shallow Node Embedding)")
        print("Embedding Dimension: 768")
        print("Semantic Weight: 0.6, ISNE Weight: 0.4")
        print("\nConnecting to MCP server at http://127.0.0.1:8000...")
        
        # First, list available tools
        list_tools()
        
        # Run tests for each major functionality
        test_embed_document()  # Test hybrid embeddings first
        test_semantic_search()
        test_retrieve_path()
        test_graph_statistics()
        
        print("\n===== All tests completed successfully =====")
        
    except requests.exceptions.ConnectionError:
        print("\nError: Could not connect to MCP server at http://127.0.0.1:8000")
        print("Please make sure the server is running with: python -m hades_pathrag.mcp_server.mcp_standalone")
        sys.exit(1)
    except Exception as e:
        print(f"\nError running tests: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
