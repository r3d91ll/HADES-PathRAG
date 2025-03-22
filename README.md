The code for the paper **"PathRAG: Pruning Graph-based Retrieval Augmented Generation with Relational Paths"**.
## Install
```bash
cd PathRAG
pip install -e .
```
## Quick Start
* You can quickly experience this project in the `v1_test.py` file.
* Set OpenAI API key in environment if using OpenAI models: `api_key="sk-...".` in the `v1_test.py` and `llm.py` file
* Prepare your retrieval document "text.txt".
* Use the following Python snippet in the "v1_text.py" file to initialize PathRAG and perform queries.
  
```python
import os
from PathRAG import PathRAG, QueryParam
from PathRAG.llm import gpt_4o_mini_complete

WORKING_DIR = "./your_working_dir"
api_key="your_api_key"
os.environ["OPENAI_API_KEY"] = api_key
base_url="https://api.openai.com/v1"
os.environ["OPENAI_API_BASE"]=base_url


if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = PathRAG(
    working_dir=WORKING_DIR,
    llm_model_func=gpt_4o_mini_complete,  
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
