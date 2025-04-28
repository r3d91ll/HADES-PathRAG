#!/usr/bin/env python3
"""
Test script for the PathRAG path ranking algorithm.

This script demonstrates the PathRAG path ranking algorithm on a sample repository,
showing how paths are ranked based on semantic relevance, path length, and edge strength.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
from tabulate import tabulate
import networkx as nx
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.isne.path_ranking import PathRanker
from src.isne.types.models import IngestDocument, DocumentRelation, RelationType
from src.db.arango_connection import ArangoConnection
from src.ingest.repository.arango_repository import ArangoRepository
from src.isne.integrations.pathrag_connector import PathRAGConnector
from src.isne.pipeline import ISNEPipeline, PipelineConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_sample_graph() -> tuple[List[IngestDocument], List[DocumentRelation]]:
    """
    Create a sample document graph for testing.
    
    Returns:
        Tuple of (documents, relations)
    """
    # Create sample documents
    documents = [
        IngestDocument(
            id="doc1",
            content="This is a sample Python module that defines basic mathematical operations.",
            source="src/math_utils.py",
            document_type="python",
            title="Math Utilities",
            embedding=[0.1, 0.2, 0.3, 0.4, 0.5]
        ),
        IngestDocument(
            id="doc2",
            content="def add(a, b):\n    \"\"\"Add two numbers and return the result.\"\"\"\n    return a + b",
            source="src/math_utils.py",
            document_type="python",
            title="add function",
            embedding=[0.2, 0.3, 0.4, 0.5, 0.6]
        ),
        IngestDocument(
            id="doc3",
            content="def subtract(a, b):\n    \"\"\"Subtract b from a and return the result.\"\"\"\n    return a - b",
            source="src/math_utils.py",
            document_type="python",
            title="subtract function",
            embedding=[0.3, 0.4, 0.5, 0.6, 0.7]
        ),
        IngestDocument(
            id="doc4",
            content="def multiply(a, b):\n    \"\"\"Multiply two numbers and return the result.\"\"\"\n    return a * b",
            source="src/math_utils.py",
            document_type="python",
            title="multiply function",
            embedding=[0.4, 0.5, 0.6, 0.7, 0.8]
        ),
        IngestDocument(
            id="doc5",
            content="def divide(a, b):\n    \"\"\"Divide a by b and return the result.\"\"\"\n    if b == 0:\n        raise ValueError(\"Cannot divide by zero\")\n    return a / b",
            source="src/math_utils.py",
            document_type="python",
            title="divide function",
            embedding=[0.5, 0.6, 0.7, 0.8, 0.9]
        ),
        IngestDocument(
            id="doc6",
            content="class Calculator:\n    \"\"\"A simple calculator class that provides basic math operations.\"\"\"\n    \n    def __init__(self):\n        self.result = 0\n    \n    def clear(self):\n        \"\"\"Reset the calculator.\"\"\"\n        self.result = 0",
            source="src/calculator.py",
            document_type="python",
            title="Calculator class",
            embedding=[0.6, 0.7, 0.8, 0.9, 1.0]
        ),
        IngestDocument(
            id="doc7",
            content="def calculate(operation, a, b):\n    \"\"\"Perform a calculation based on the operation name.\"\"\"\n    if operation == 'add':\n        return add(a, b)\n    elif operation == 'subtract':\n        return subtract(a, b)\n    elif operation == 'multiply':\n        return multiply(a, b)\n    elif operation == 'divide':\n        return divide(a, b)\n    else:\n        raise ValueError(f\"Unknown operation: {operation}\")",
            source="src/calculator.py",
            document_type="python",
            title="calculate function",
            embedding=[0.7, 0.8, 0.9, 1.0, 1.1]
        ),
        IngestDocument(
            id="doc8",
            content="# Math Utilities\n\nThis module provides basic mathematical operations for the calculator application.",
            source="src/README.md",
            document_type="markdown",
            title="Math Utilities README",
            embedding=[0.8, 0.9, 1.0, 1.1, 1.2]
        ),
        IngestDocument(
            id="doc9",
            content="# Calculator Application\n\nThis application demonstrates a simple calculator with basic math operations.",
            source="README.md",
            document_type="markdown",
            title="Main README",
            embedding=[0.9, 1.0, 1.1, 1.2, 1.3]
        ),
        IngestDocument(
            id="doc10",
            content="import math_utils\n\ndef advanced_operations(a, b):\n    \"\"\"Perform advanced math operations.\"\"\"\n    result = {\n        'add': math_utils.add(a, b),\n        'subtract': math_utils.subtract(a, b),\n        'multiply': math_utils.multiply(a, b),\n        'divide': math_utils.divide(a, b),\n        'power': math_utils.power(a, b)\n    }\n    return result",
            source="src/advanced_calculator.py",
            document_type="python",
            title="Advanced Calculator",
            embedding=[1.0, 1.1, 1.2, 1.3, 1.4]
        )
    ]
    
    # Create relations between documents
    relations = [
        # Module contains functions
        DocumentRelation(
            source_id="doc1",
            target_id="doc2",
            relation_type=RelationType.CONTAINS,
            weight=1.0
        ),
        DocumentRelation(
            source_id="doc1",
            target_id="doc3",
            relation_type=RelationType.CONTAINS,
            weight=1.0
        ),
        DocumentRelation(
            source_id="doc1",
            target_id="doc4",
            relation_type=RelationType.CONTAINS,
            weight=1.0
        ),
        DocumentRelation(
            source_id="doc1",
            target_id="doc5",
            relation_type=RelationType.CONTAINS,
            weight=1.0
        ),
        # Calculator class contains methods
        DocumentRelation(
            source_id="doc6",
            target_id="doc7",
            relation_type=RelationType.CONTAINS,
            weight=1.0
        ),
        # Calculate function uses math functions
        DocumentRelation(
            source_id="doc7",
            target_id="doc2",
            relation_type=RelationType.REFERENCES,
            weight=0.8
        ),
        DocumentRelation(
            source_id="doc7",
            target_id="doc3",
            relation_type=RelationType.REFERENCES,
            weight=0.8
        ),
        DocumentRelation(
            source_id="doc7",
            target_id="doc4",
            relation_type=RelationType.REFERENCES,
            weight=0.8
        ),
        DocumentRelation(
            source_id="doc7",
            target_id="doc5",
            relation_type=RelationType.REFERENCES,
            weight=0.8
        ),
        # READMEs document the code
        DocumentRelation(
            source_id="doc8",
            target_id="doc1",
            relation_type=RelationType.DOCUMENTS,
            weight=0.9
        ),
        DocumentRelation(
            source_id="doc9",
            target_id="doc6",
            relation_type=RelationType.DOCUMENTS,
            weight=0.9
        ),
        # Advanced calculator imports math utils
        DocumentRelation(
            source_id="doc10",
            target_id="doc1",
            relation_type=RelationType.IMPORTS,
            weight=0.7
        ),
        # Advanced calculator uses math functions
        DocumentRelation(
            source_id="doc10",
            target_id="doc2",
            relation_type=RelationType.REFERENCES,
            weight=0.6
        ),
        DocumentRelation(
            source_id="doc10",
            target_id="doc3",
            relation_type=RelationType.REFERENCES,
            weight=0.6
        ),
        DocumentRelation(
            source_id="doc10",
            target_id="doc4",
            relation_type=RelationType.REFERENCES,
            weight=0.6
        ),
        DocumentRelation(
            source_id="doc10",
            target_id="doc5",
            relation_type=RelationType.REFERENCES,
            weight=0.6
        )
    ]
    
    return documents, relations


def visualize_graph(documents: List[IngestDocument], relations: List[DocumentRelation], output_path: str) -> None:
    """
    Visualize the document graph.
    
    Args:
        documents: List of documents
        relations: List of relations
        output_path: Path to save the visualization
    """
    # Create graph
    graph = nx.DiGraph()
    
    # Add nodes
    for doc in documents:
        graph.add_node(doc.id, title=doc.title, type=doc.document_type)
    
    # Add edges
    for rel in relations:
        graph.add_edge(rel.source_id, rel.target_id, type=rel.relation_type.value, weight=rel.weight)
    
    # Create node positions using spring layout
    pos = nx.spring_layout(graph, seed=42)
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Draw nodes
    node_types = {doc.id: doc.document_type for doc in documents}
    colors = {
        "python": "skyblue",
        "markdown": "lightgreen"
    }
    node_colors = [colors.get(node_types.get(node, "unknown"), "gray") for node in graph.nodes]
    
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=700, alpha=0.8)
    
    # Draw edges
    edge_types = {(rel.source_id, rel.target_id): rel.relation_type.value for rel in relations}
    edge_colors = {
        "contains": "blue",
        "references": "red",
        "documents": "green",
        "imports": "purple"
    }
    edges = list(graph.edges())
    edge_colors_list = [edge_colors.get(edge_types.get((u, v), "unknown"), "gray") for u, v in edges]
    
    nx.draw_networkx_edges(graph, pos, edgelist=edges, edge_color=edge_colors_list, width=2, alpha=0.7)
    
    # Draw labels
    labels = {doc.id: doc.title for doc in documents}
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=10, font_weight="bold")
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='skyblue', markersize=15, label='Python'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', markersize=15, label='Markdown'),
        Line2D([0], [0], color='blue', lw=4, label='Contains'),
        Line2D([0], [0], color='red', lw=4, label='References'),
        Line2D([0], [0], color='green', lw=4, label='Documents'),
        Line2D([0], [0], color='purple', lw=4, label='Imports')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.title("Document Graph Visualization")
    plt.axis('off')
    
    # Save figure
    plt.savefig(output_path)
    plt.close()
    
    logger.info(f"Graph visualization saved to {output_path}")


def test_path_ranking(output_dir: str) -> None:
    """
    Test the PathRAG path ranking algorithm.
    
    Args:
        output_dir: Directory to save results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create sample graph
    documents, relations = create_sample_graph()
    
    # Visualize graph
    visualize_graph(documents, relations, os.path.join(output_dir, "document_graph.png"))
    
    # Create path ranker
    path_ranker = PathRanker(
        semantic_weight=0.7,
        path_length_weight=0.1,
        edge_strength_weight=0.2,
        max_path_length=5,
        max_paths=20
    )
    
    # Test queries
    test_queries = [
        "How to divide two numbers in Python?",
        "How does the calculator use math functions?",
        "What are the basic math operations?",
        "How to perform advanced calculations?"
    ]
    
    # Create mock embedding function for queries
    def mock_query_embedding(query: str) -> List[float]:
        """Create a mock embedding for a query."""
        if "divide" in query.lower():
            return [0.5, 0.6, 0.7, 0.8, 0.9]  # Similar to divide function
        elif "calculator" in query.lower():
            return [0.6, 0.7, 0.8, 0.9, 1.0]  # Similar to Calculator class
        elif "math" in query.lower():
            return [0.1, 0.2, 0.3, 0.4, 0.5]  # Similar to math utils module
        elif "advanced" in query.lower():
            return [1.0, 1.1, 1.2, 1.3, 1.4]  # Similar to advanced calculator
        else:
            return [0.5, 0.5, 0.5, 0.5, 0.5]  # Neutral
    
    # Test path ranking for each query
    for i, query in enumerate(test_queries):
        logger.info(f"Testing query: {query}")
        
        # Create mock query embedding
        query_embedding = mock_query_embedding(query)
        
        # Build graph
        graph = path_ranker.build_graph(documents, relations)
        
        # Find best source node based on semantic similarity
        source_node = None
        max_sim = -1.0
        
        for doc in documents:
            if doc.embedding is not None:
                sim = path_ranker.compute_semantic_similarity(query_embedding, doc.embedding)
                if sim > max_sim:
                    max_sim = sim
                    source_node = doc.id
        
        # Get all node IDs
        all_nodes = [doc.id for doc in documents]
        
        # Rank paths
        ranked_paths = path_ranker.rank_paths(
            query_embedding=query_embedding,
            source_node=source_node,
            target_nodes=all_nodes,
            graph=graph
        )
        
        # Add node details
        enriched_paths = path_ranker.get_node_details(ranked_paths, graph)
        
        # Save results
        result = {
            "query": query,
            "source_node": source_node,
            "paths": enriched_paths
        }
        
        with open(os.path.join(output_dir, f"query_{i+1}_results.json"), "w") as f:
            json.dump(result, f, indent=2)
        
        # Print summary table
        print("\nQuery:", query)
        print("Source node:", source_node, "(", next((doc.title for doc in documents if doc.id == source_node), "Unknown"), ")")
        print("\nTop paths:")
        
        table_data = []
        for j, path in enumerate(enriched_paths[:5]):  # Show top 5 paths
            path_str = " → ".join([node["title"] for node in path["nodes"]])
            table_data.append([
                j + 1,
                f"{path['score']:.4f}",
                f"{path['semantic_score']:.4f}",
                f"{path['path_length_score']:.4f}",
                f"{path['edge_strength']:.4f}",
                path_str
            ])
        
        headers = ["Rank", "Score", "Semantic", "Length", "Edge", "Path"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        print("\n" + "-" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Test PathRAG path ranking algorithm")
    parser.add_argument("--output-dir", type=str, default="./path_ranking_results", help="Directory to save results")
    args = parser.parse_args()
    
    test_path_ranking(args.output_dir)


if __name__ == "__main__":
    main()
