#!/bin/bash
# Setup script for HADES-PathRAG MCP server with Qwen2.5-128k integration
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

echo "🔧 Setting up HADES-PathRAG MCP server with Qwen2.5-128k..."

# Activate and setup virtual environment
if [ -d ".venv" ]; then
    echo "🔧 Using existing virtual environment..."
else
    echo "🔧 Creating virtual environment..."
    python3 -m venv .venv
fi

# Make sure Python dependencies are installed
if [ -f "requirements.txt" ]; then
    echo "📦 Installing Python dependencies..."
    .venv/bin/pip install -r requirements.txt
fi

# Ensure Ollama is running
if ! pgrep -x "ollama" > /dev/null; then
    echo "⚠️ Ollama service is not running! Starting it..."
    ollama serve &
    sleep 5
fi

# Check if our Qwen model is available
if ! ollama list | grep -q "qwen2.5-128k"; then
    echo "🤖 Creating Qwen2.5-128k model from Modelfile..."
    ollama create qwen2.5-128k -f Modelfile.qwen2.5
fi

# Update the Windsurf IDE config to use our MCP server
MCP_CONFIG_PATH="$HOME/.codeium/windsurf/mcp_config.json"
if [ -f "$MCP_CONFIG_PATH" ]; then
    echo "🔄 Updating Windsurf MCP configuration..."
    # Create a backup of the existing config
    cp "$MCP_CONFIG_PATH" "${MCP_CONFIG_PATH}.bak"
    
    # Update the config to point to our repository
    # This is a simplified approach - in production you'd use jq or similar
    sed -i "s|\"command\": \".*\"|\"command\": \"$REPO_ROOT/.venv/bin/python\"|g" "$MCP_CONFIG_PATH"
    sed -i "s|\"cwd\": \".*ML-Lab/Heuristic-Adaptive-Data-Extrapolation-System-HADES.*\"|\"cwd\": \"$REPO_ROOT\"|g" "$MCP_CONFIG_PATH"
    sed -i "s|\"PYTHONPATH\": \".*ML-Lab/Heuristic-Adaptive-Data-Extrapolation-System-HADES.*\"|\"PYTHONPATH\": \"$REPO_ROOT\"|g" "$MCP_CONFIG_PATH"
    
    echo "✅ MCP configuration updated to use HADES-PathRAG"
else
    echo "⚠️ Windsurf MCP configuration not found at $MCP_CONFIG_PATH"
fi

# Create a .env file with necessary settings if it doesn't exist
if [ ! -f "$REPO_ROOT/.env" ]; then
    echo "🔑 Creating .env file with default settings..."
    cat > "$REPO_ROOT/.env" << EOF
# HADES-PathRAG Environment Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen2.5-128k
OLLAMA_EMBED_MODEL=nomic-embed-text
OLLAMA_TIMEOUT=120
OLLAMA_TEMPERATURE=0.65
OLLAMA_TOP_P=0.95
OLLAMA_TOP_K=30
OLLAMA_REPEAT_PENALTY=1.1
OLLAMA_MIN_P=0.0
OLLAMA_MAX_TOKENS=4096

# Default to Ollama as LLM provider
LLM_PROVIDER=ollama

# ArangoDB Configuration
ARANGO_HOST=http://localhost:8529
ARANGO_DB=pathrag
ARANGO_USER=root
ARANGO_PASSWORD=
EOF
    echo "✅ Created .env file with default settings"
else
    echo "ℹ️ Using existing .env file"
fi

echo "🚀 HADES-PathRAG MCP server setup complete!"
echo ""
echo "To start the MCP server, run:"
echo "cd $REPO_ROOT && python -m src.mcp.server"
echo ""
echo "To use it with Windsurf, restart the IDE after running this setup script."
