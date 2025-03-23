# HADES-PathRAG

Enhanced implementation of **PathRAG: Pruning Graph-based Retrieval Augmented Generation with Relational Paths** with XnX notation, ArangoDB integration, and **native Ollama support** for seamless local LLM inference using your existing Ollama installation.

## Features

- 🔍 **PathRAG Implementation**: Efficient graph-based RAG that prunes retrieval paths
- 🧠 **Native Ollama Integration**: Works directly with locally installed Ollama service for optimized performance
- 🔗 **XnX Notation**: Enhanced relationship representation in knowledge graphs
- 📊 **ArangoDB Support**: Scalable graph database backend for enterprise use
- 🔄 **HADES Recursive Architecture**: Self-improving AI system framework
- 🚀 **MCP Server**: Model Context Protocol for IDE integration

## Install

```bash
# Clone the repository
git clone https://github.com/r3d91ll/HADES-PathRAG.git
cd HADES-PathRAG

# Install dependencies
pip install -e .

# Use your existing Ollama installation
# Verify Ollama is running as a system service:
systemctl status ollama

# Or install Ollama if needed (automatically sets up as a system service on Linux):
# curl -fsSL https://ollama.com/install.sh | sh
```
## Quick Start
* You can quickly experience this project in the `v1_test.py` file.
* Set OpenAI API key in environment if using OpenAI models: `api_key="sk-...".` in the `v1_test.py` and `llm.py` file
* Prepare your retrieval document "text.txt".
* Use the following Python snippet in the "v1_text.py" file to initialize PathRAG and perform queries.
  
```python
import os
from pathrag import PathRAG, QueryParam

# For using Ollama (no API keys needed)
from pathrag.llm import ollama_model_complete

# For using OpenAI (if preferred)
# from pathrag.llm import openai_complete
# os.environ["OPENAI_API_KEY"] = "your_api_key"

WORKING_DIR = "./your_working_dir"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# Initialize with Ollama as the LLM provider
rag = PathRAG(
    working_dir=WORKING_DIR,
    llm_model_func=lambda prompt, **kwargs: ollama_model_complete(
        prompt=prompt,
        hashing_kv={"global_config": {"llm_model_name": "llama3"}},
        host="http://localhost:11434"
    )
)

data_file="./text.txt"
question="your_question"
with open(data_file) as f:
    rag.insert(f.read())

print(rag.query(question, param=QueryParam(mode="hybrid")))
```
## Parameter modification
You can adjust the relevant parameters in the `base.py` and `operate.py` files.

## Batch Insert
```python
import os
folder_path = "your_folder_path"  

txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
for file_name in txt_files:
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, "r", encoding="utf-8") as file:
        rag.insert(file.read())
```

## Cite
Please cite our paper if you use this code in your own work:
```python
@article{chen2025pathrag,
  title={PathRAG: Pruning Graph-based Retrieval Augmented Generation with Relational Paths},
  author={Chen, Boyu and Guo, Zirui and Yang, Zidan and Chen, Yuluo and Chen, Junze and Liu, Zhenghao and Shi, Chuan and Yang, Cheng},
  journal={arXiv preprint arXiv:2502.14902},
  year={2025}
}
```

## HADES: XnX-Enhanced PathRAG Implementation

This fork contains an enhanced version of PathRAG that integrates XnX notation, ArangoDB, and Ollama for the HADES project.

### What is HADES?

Wondering about our project name? HADES isn't just the Greek god of the underworld - it's a carefully crafted backronym:

- **H**euristic (XnX notation for weighted path tuning)
- **A**daptive (PathRAG implementation with relationship pruning)
- **D**ata (ArangoDB knowledge graph storage)
- **E**xtrapolation (Ollama model inference)
- **S**ystems (because we needed an S!)

> *Note to critics: Yes, we crafted the name first and built the technology to match it. That's just how cool we are.*

### Key Enhancements

- **XnX Notation**: Added weight, direction, and distance parameters to fine-tune path retrieval
- **ArangoDB Integration**: Optimized graph storage with combined vector and graph operations
- **MCP Server**: REST API with entity-relationship model clarity
- **Ollama Integration**: Local LLM inference for improved privacy and control
- **Web Interface**: Modern UI for interacting with the system

Run `python start_server.py` to launch the enhanced PathRAG with MCP and Ollama integration.
