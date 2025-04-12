#!/usr/bin/env python
"""
Test client for MCP server.

This script tests the MCP server by sending requests and receiving responses
using the Model Context Protocol.
"""
import asyncio
import json
import logging
import sys
import os
from pathlib import Path

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import aiohttp
import websockets


async def test_http_client() -> None:
    """Test the MCP server using HTTP transport."""
    print("\n--- Testing HTTP Transport ---")
    async with aiohttp.ClientSession() as session:
        # Step 1: List available tools
        print("\nListing available tools...")
        async with session.post(
            "http://localhost:8000/mcp/jsonrpc",
            json={
                "type": "request",
                "id": "1",
                "method": "tools/list",
                "params": {}
            },
            headers={
                "Content-Type": "application/json"
            }
        ) as response:
            if response.status == 200:
                result = await response.json()
                print(f"Available tools: {json.dumps(result, indent=2)}")
                
                # Extract tool names for further testing
                tools = []
                if "result" in result and "tools" in result["result"]:
                    tools = [tool["name"] for tool in result["result"]["tools"]]
                    
                # Step 2: Test a tool (if available)
                if "retrieve_path" in tools:
                    print("\nTesting 'retrieve_path' tool...")
                    async with session.post(
                        "http://localhost:8000/mcp/jsonrpc",
                        json={
                            "type": "request",
                            "id": "2",
                            "method": "tools/call",
                            "params": {
                                "name": "retrieve_path",
                                "arguments": {
                                    "query": "How does DNA replication work?",
                                    "max_length": 3
                                }
                            }
                        },
                        headers={
                            "Content-Type": "application/json"
                        }
                    ) as tool_response:
                        if tool_response.status == 200:
                            tool_result = await tool_response.json()
                            print(f"Tool response: {json.dumps(tool_result, indent=2)}")
                        else:
                            print(f"Error calling tool: {await tool_response.text()}")
                
                # Step 3: Test another tool (if available)
                if "semantic_search" in tools:
                    print("\nTesting 'semantic_search' tool...")
                    async with session.post(
                        "http://localhost:8000/mcp/jsonrpc",
                        json={
                            "type": "request",
                            "id": "3",
                            "method": "tools/call",
                            "params": {
                                "name": "semantic_search",
                                "arguments": {
                                    "query": "machine learning algorithms",
                                    "top_k": 3
                                }
                            }
                        },
                        headers={
                            "Content-Type": "application/json"
                        }
                    ) as tool_response:
                        if tool_response.status == 200:
                            tool_result = await tool_response.json()
                            print(f"Tool response: {json.dumps(tool_result, indent=2)}")
                        else:
                            print(f"Error calling tool: {await tool_response.text()}")
            else:
                print(f"Error listing tools: {await response.text()}")


async def main() -> None:
    """Run the test client."""
    print("MCP Server Test Client")
    print("=====================")
    
    # Test HTTP client
    await test_http_client()


if __name__ == "__main__":
    asyncio.run(main())
