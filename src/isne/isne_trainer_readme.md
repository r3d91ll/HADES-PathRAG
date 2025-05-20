# ISNE Trainer Module

## Overview

The ISNE Trainer Module provides a complete implementation for training the Inductive Shallow Node Embedding (ISNE) model as part of the HADES-PathRAG system. This module bridges the output from the document processing pipeline to the ISNE model training process, enabling enhanced document embeddings through graph-based learning.

## Core Components

### ISNETrainingOrchestrator

The primary class responsible for coordinating the end-to-end ISNE training workflow:

- **Input Processing**: Takes processed documents with ModernBERT embeddings from the document processing pipeline
- **Graph Construction**: Builds a document graph from the embeddings and document relationships
- **ISNE Model Training**: Trains the ISNE model on the constructed graph
- **Model Evaluation**: Evaluates the trained model using appropriate metrics
- **Model Persistence**: Saves trained models for later use in inference

### Configuration System

Extends the existing pipeline configuration system to include ISNE training parameters:

- **Training Parameters**: Epochs, learning rate, batch size, model architecture settings
- **Resource Allocation**: CPU/GPU usage settings, aligned with the document processing pipeline
- **File Handling**: Input/output directory configurations

### Workflow Integration

Seamlessly integrates with the existing document processing pipeline:

1. **Pipeline Output Consumption**: Reads the output from the multiprocessing pipeline test
2. **Document Graph Creation**: Converts document relationships into a graph structure
3. **Training Loop Execution**: Implements efficient training with progress tracking
4. **Model Serialization**: Stores models with version history

## Usage

### Basic Command-line Usage

```bash
python -m src.isne.trainer.training_orchestrator \
    --input-dir ./test-output/pipeline-mp-test \
    --model-output-dir ./models/isne \
    --epochs 50 \
    --device cpu
```

### Programmatic Usage

```python
from src.isne.trainer.training_orchestrator import ISNETrainingOrchestrator

# Initialize the orchestrator
orchestrator = ISNETrainingOrchestrator(
    input_dir="./test-output/pipeline-mp-test",
    model_output_dir="./models/isne",
    config_override={
        "training": {
            "epochs": 50,
            "learning_rate": 0.001,
            "batch_size": 32
        }
    }
)

# Run the training process
training_results = orchestrator.train()

# Get trained model for inference
model = orchestrator.load_model()
```

### Integration with Pipeline Test

```bash
# First run the document processing pipeline
python -m tests.integration.pipeline_multiprocess_test --num-files 20

# Then train the ISNE model on the processed documents
python -m src.isne.trainer.training_orchestrator \
    --input-dir ./test-output/pipeline-mp-test
```

## Implementation

### Graph Construction

The trainer converts document chunks and their relationships into a PyTorch Geometric graph structure:

1. **Node Features**: Document chunk embeddings from ModernBERT
2. **Edge Construction**: Relationships between chunks based on:
   - Sequential proximity within documents
   - Semantic similarity between chunks
   - Explicit relationships like citations or references

### Training Procedure

The training procedure follows the approach defined in the ISNE paper:

1. **Batching**: Mini-batch training with neighborhood sampling
2. **Loss Computation**: Multi-objective loss combining:
   - Feature preservation loss (maintaining original semantic information)
   - Structural preservation loss (preserving graph structure)
   - Contrastive loss (differentiating between related and unrelated chunks)
3. **Optimization**: Adam optimizer with learning rate scheduling

### Evaluation Metrics

The trainer evaluates model performance using:

- **Embedding Quality**: Cosine similarity between related documents
- **Link Prediction**: Ability to predict connections between documents
- **Ablation Studies**: Measuring contribution of different loss components

## Benchmarks

The module includes performance benchmarks for:

- **Training Time**: Measurement of training duration across different document set sizes
- **Embedding Quality**: Comparison of ISNE-enhanced embeddings vs. original embeddings
- **Memory Usage**: Profiling of memory requirements during training

## Testing

Following the project's standard protocol, this module includes comprehensive testing:

- **Unit Tests**: Individual components tested with ≥85% coverage
- **Integration Tests**: End-to-end tests with real document examples
- **Performance Tests**: Measurement of training efficiency

## Future Improvements

Planned enhancements for this module include:

1. **Distributed Training**: Support for training across multiple nodes
2. **Dynamic Graph Updates**: Incremental training for evolving document collections
3. **Advanced Relationship Extraction**: More sophisticated methods for identifying document relationships
4. **Hyperparameter Optimization**: Automated tuning of model parameters
