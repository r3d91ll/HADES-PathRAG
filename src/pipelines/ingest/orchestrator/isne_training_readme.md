# ISNE Training Module Documentation

## Overview

The ISNE (Inductive Shallow Node Embedding) Training Module provides specialized functionality for training the ISNE model on text documents. This module processes PDF files in a specified directory, creates document embeddings, builds a document graph, and trains the ISNE model to enhance these embeddings with graph structural information.

## Core Components

### TextTrainingPipeline Class

The primary class that orchestrates the complete training workflow:

- **Document Processing**: Transforms source documents into normalized text
- **Chunking**: Splits documents into semantic chunks
- **Embedding**: Computes vector embeddings for chunks using ModernBERT
- **Graph Construction**: Builds a graph structure linking chunks both sequentially and semantically
- **ISNE Training**: Trains the ISNE model on the constructed graph
- **Model Saving**: Persists the trained model for later use

## Usage

### Command-line Interface

```bash
python -m src.pipelines.ingest.orchestrator.isne_training_text \
    --input_dir /path/to/pdf/directory \
    --output_dir ./test-output/isne-training \
    --model_dir ./models/isne \
    --file_pattern "*.pdf" \
    --epochs 50 \
    --learning_rate 0.001 \
    --hidden_dim 256 \
    --device cpu
```

### Programmatic Usage

```python
from src.pipelines.ingest.orchestrator.isne_training_text import run_training_pipeline
import asyncio

asyncio.run(run_training_pipeline(
    input_dir="/path/to/pdf/directory",
    output_dir="./test-output/isne-training",
    model_output_dir="./models/isne",
    training_options={
        "epochs": 50,
        "learning_rate": 0.001,
        "hidden_dim": 256
    },
    file_pattern="*.pdf",
    save_intermediate_results=True
))
```

## Training Process

1. **Document Collection**: All PDF files in the input directory are identified
2. **Document Processing**: Each document is processed through:
   - Text extraction and normalization
   - Semantic chunking
   - Embedding computation
3. **Graph Construction**:
   - Chunks within documents are connected sequentially
   - Similarity-based connections are created between related chunks
4. **ISNE Training**:
   - The model learns to enhance embeddings based on graph structure
   - Training uses a combination of reconstruction loss and structural preservation
   - The model is periodically checkpointed and evaluated
5. **Model Persistence**:
   - The best model is saved for later use
   - Intermediate results are saved at each stage for analysis

## Configuration Options

### Chunking Options
- `max_tokens`: Maximum token count per chunk (default: 1024)
- `doc_type`: Document type for chunking strategy (default: "academic_pdf")

### Embedding Options
- `adapter_name`: Embedding adapter to use (default: "modernbert")
- `model_name`: Specific model for the adapter (default: "answerdotai/ModernBERT-base")
- `device`: Computation device (default: "cpu")
- `normalize_embeddings`: Whether to L2-normalize embeddings (default: True)

### Training Options
- `batch_size`: Training batch size (default: 32)
- `learning_rate`: Initial learning rate (default: 0.001)
- `epochs`: Number of training epochs (default: 50)
- `hidden_dim`: Hidden layer dimension (default: 256)
- `num_layers`: Number of ISNE layers (default: 2)
- `num_heads`: Number of attention heads (default: 8)
- `dropout`: Dropout probability (default: 0.1)
- `early_stopping_patience`: Epochs without improvement before stopping (default: 10)
- `checkpoint_interval`: Save frequency in epochs (default: 5)

## Testing and Performance

For testing module performance, use the following command to train on a small set of documents:

```bash
python -m src.pipelines.ingest.orchestrator.isne_training_text \
    --input_dir ./data/test-papers \
    --output_dir ./test-output/isne-test \
    --epochs 10 \
    --device cpu
```

## Future Improvements

- Integration with GPU accelerated embedding extraction via vLLM
- Support for distributed training on larger document collections
- Implementation of more sophisticated document relation extraction
- Dynamic adjustment of graph construction parameters
