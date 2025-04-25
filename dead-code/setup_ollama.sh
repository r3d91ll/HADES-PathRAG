#!/bin/bash
# HADES-PathRAG Ollama Setup Script
# This script helps set up Ollama for use with HADES-PathRAG

set -e  # Exit on error

# Set text colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=====================================${NC}"
echo -e "${BLUE}HADES-PathRAG Ollama Setup Assistant${NC}"
echo -e "${BLUE}=====================================${NC}"
echo ""

# Check if Ollama is already installed
if command -v ollama &> /dev/null; then
    echo -e "${GREEN}✓ Ollama is already installed.${NC}"
    OLLAMA_VERSION=$(ollama --version 2>&1)
    echo -e "  Current version: ${YELLOW}$OLLAMA_VERSION${NC}"
else
    echo -e "${YELLOW}! Ollama is not installed. Installing now...${NC}"
    
    # Check OS type
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo -e "  Detected Linux OS. Installing Ollama..."
        curl -fsSL https://ollama.com/install.sh | sh
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo -e "  Detected macOS. Installing Ollama..."
        curl -fsSL https://ollama.com/install.sh | sh
    else
        echo -e "${RED}✗ Unsupported OS. Please install Ollama manually from https://ollama.com/download${NC}"
        echo "  After installation, run this script again."
        exit 1
    fi
    
    echo -e "${GREEN}✓ Ollama installed successfully.${NC}"
fi

# Check if Ollama service is running
if curl -s http://localhost:11434/api/tags &> /dev/null; then
    echo -e "${GREEN}✓ Ollama service is running.${NC}"
else
    echo -e "${YELLOW}! Ollama service is not running. Starting now...${NC}"
    echo -e "  Starting Ollama service in the background..."
    
    # Start Ollama service in the background
    nohup ollama serve > /tmp/ollama.log 2>&1 &
    
    # Wait for service to start
    echo -n "  Waiting for Ollama service to start"
    for i in {1..10}; do
        if curl -s http://localhost:11434/api/tags &> /dev/null; then
            echo -e "\n${GREEN}✓ Ollama service started successfully.${NC}"
            break
        fi
        echo -n "."
        sleep 2
        
        if [ $i -eq 10 ]; then
            echo -e "\n${RED}✗ Failed to start Ollama service. Check logs at /tmp/ollama.log${NC}"
            exit 1
        fi
    done
fi

# Check for required models
echo -e "\n${BLUE}Checking for required models...${NC}"

# Function to check if model exists and pull if needed
check_and_pull_model() {
    local model_name="$1"
    if ollama list | grep -q "$model_name"; then
        echo -e "${GREEN}✓ Model '$model_name' is already pulled.${NC}"
    else
        echo -e "${YELLOW}! Model '$model_name' not found. Pulling now...${NC}"
        echo -e "  This may take some time depending on your internet connection and the model size."
        ollama pull "$model_name"
        echo -e "${GREEN}✓ Successfully pulled model '$model_name'.${NC}"
    fi
}

# Check for LLM model (llama3)
check_and_pull_model "llama3"

# Check for embedding model (nomic-embed-text)
check_and_pull_model "nomic-embed-text"

# Create/update .env file with Ollama settings
echo -e "\n${BLUE}Setting up environment configuration...${NC}"

# Check if .env file exists
if [ -f ".env" ]; then
    echo -e "${YELLOW}! Found existing .env file. Creating backup at .env.bak${NC}"
    cp .env .env.bak
else
    echo -e "  Creating new .env file from template..."
    cp .env.template .env 2>/dev/null || echo -e "  ${YELLOW}No .env.template found, creating new .env file...${NC}"
fi

# Update or add Ollama configuration to .env
{
    grep -v "^LLM_PROVIDER\|^OLLAMA_" .env 2>/dev/null || true
    echo ""
    echo "# LLM Provider"
    echo "LLM_PROVIDER=ollama"
    echo ""
    echo "# Ollama Settings"
    echo "OLLAMA_HOST=http://localhost:11434"
    echo "OLLAMA_MODEL=llama3"
    echo "OLLAMA_EMBED_MODEL=nomic-embed-text"
    echo "OLLAMA_TIMEOUT=60"
    echo "OLLAMA_TEMPERATURE=0.7"
    echo "OLLAMA_TOP_P=0.9"
    echo "OLLAMA_TOP_K=40"
    echo "OLLAMA_MAX_TOKENS=2048"
} > .env.new

mv .env.new .env
echo -e "${GREEN}✓ Environment configuration updated.${NC}"

# Verify and test the setup
echo -e "\n${BLUE}Testing Ollama integration...${NC}"
if [ -f "examples/ollama_example.py" ]; then
    echo -e "  Would you like to test the Ollama integration now? (y/n)"
    read -r run_test
    
    if [[ "$run_test" =~ ^[Yy]$ ]]; then
        echo -e "  Running Ollama test example..."
        python examples/ollama_example.py
        echo -e "\n${GREEN}✓ Test complete.${NC}"
    else
        echo -e "  Skipping test. You can test later with: python examples/ollama_example.py"
    fi
else
    echo -e "${YELLOW}! Test script not found. You can create a test script to verify the setup.${NC}"
fi

echo -e "\n${GREEN}=====================================${NC}"
echo -e "${GREEN}Ollama setup complete!${NC}"
echo -e "${GREEN}=====================================${NC}"
echo -e "You can now use Ollama as your LLM provider for HADES-PathRAG."
echo -e "For more information, see: docs/integration/ollama_setup.md"
echo ""
