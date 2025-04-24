"""
PDF pre-processor that extracts hierarchical document structure.

This processor demonstrates the "turtles all the way down" pattern by extracting:
- Document structure (sections, subsections)
- Figures with captions
- Text content with references to figures
- All properly linked in a hierarchical graph
"""
from typing import Dict, Any, List, Optional, Tuple
import os
import io
import re
import uuid
from pathlib import Path

try:
    import fitz  # type: ignore  # PyMuPDF
except ImportError:
    fitz = None

from ..models.graph_models import (
    DocumentNode, TextNode, ImageNode, HierarchicalGraph
)
from .base_pre_processor import BasePreProcessor


class PDFPreProcessor(BasePreProcessor):
    """
    Pre-processor for PDF documents that extracts hierarchical structure.
    
    Requires PyMuPDF (install with: pip install pymupdf)
    """
    
    def __init__(self, extract_images: bool = True, extract_structure: bool = True):
        """
        Initialize the PDF pre-processor.
        
        Args:
            extract_images: Whether to extract images from the PDF
            extract_structure: Whether to extract document structure
        """
        if fitz is None:
            raise ImportError(
                "PyMuPDF is required for PDFPreProcessor. "
                "Install with: pip install pymupdf"
            )
        
        self.extract_images = extract_images
        self.extract_structure = extract_structure
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a PDF file, extracting hierarchical structure.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Hierarchical document graph
        """
        doc = fitz.open(file_path)
        
        # Create root document node
        document_node = DocumentNode(
            title=Path(file_path).stem,
            source=file_path,
            format="pdf",
            metadata={
                "page_count": len(doc),
                "author": doc.metadata.get("author", ""),
                "creator": doc.metadata.get("creator", ""),
                "producer": doc.metadata.get("producer", ""),
                "creation_date": doc.metadata.get("creationDate", "")
            }
        )
        
        # Create hierarchical graph with document as root
        graph = HierarchicalGraph(root_node=document_node)
        
        # Extract and process TOC for structure (optional)
        toc_structure: List[Dict[str, Any]] = self._extract_toc_structure(doc)
        section_nodes: Dict[str, TextNode] = {}
        
        if self.extract_structure and toc_structure:
            # Create section nodes from TOC
            for section in toc_structure:
                section_node = TextNode(
                    title=section["title"],
                    type="section",
                    metadata={
                        "level": section["level"],
                        "page_number": section["page_number"]
                    }
                )
                # Add section to graph under its parent
                if section["parent_id"] is None:
                    graph.add_child(document_node, section_node)
                else:
                    parent_id = section["parent_id"]
                    if parent_id in section_nodes:
                        graph.add_child(section_nodes[parent_id], section_node)
                    else:
                        # Fallback to document if parent not found
                        graph.add_child(document_node, section_node)
                
                # Store node for potential children
                section_nodes[section["id"]] = section_node
        
        # Process each page
        for page_idx, page in enumerate(doc):
            # Extract text content
            text_content = page.get_text()
            
            # Create page node
            page_node = TextNode(
                title=f"Page {page_idx + 1}",
                type="page",
                content=text_content,
                metadata={
                    "page_number": page_idx + 1,
                    "width": page.rect.width,
                    "height": page.rect.height
                }
            )
            
            # Add page to document
            graph.add_child(document_node, page_node)
            
            # Extract images if requested
            if self.extract_images:
                image_list: List[Any] = page.get_images(full=True)
                
                for img_idx, img_info in enumerate(image_list):
                    try:
                        xref = img_info[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        
                        # Try to identify a caption near the image
                        caption = self._find_image_caption(text_content, img_idx + 1)
                        
                        # Create image node
                        image_node = ImageNode(
                            title=caption or f"Figure {img_idx + 1}",
                            caption=caption,
                            metadata={
                                "page_number": page_idx + 1,
                                "width": base_image.get("width"),
                                "height": base_image.get("height"),
                                "xref": xref,
                                "extension": base_image.get("ext", "")
                            }
                        )
                        
                        # Add image to page node
                        graph.add_child(page_node, image_node, "contains", 1.0)
                        
                        # TODO: Store image binary data separately (S3, file system, etc.)
                        
                    except Exception as e:
                        # Skip any problematic images
                        continue
            
            # If we have found text that references figures (e.g., "see Fig. 1"),
            # create reference edges between text nodes and figure nodes
            figure_refs: List[str] = self._extract_figure_references(text_content)
            for ref in figure_refs:
                # Find an image node with matching title/number
                for node in graph.nodes:
                    if isinstance(node, ImageNode) and node.title is not None and ref in node.title:
                        # Create reference edge from page to image
                        graph.add_edge(page_node, node, "references", 0.8)
        
        # Close the document
        doc.close()
        
        return {
            "graph": graph.to_dict(),
            "file_path": file_path
        }
    
    def _extract_toc_structure(self, doc: Any) -> List[Dict[str, Any]]:
        """Extract table of contents structure."""
        toc = doc.get_toc()
        if not toc:
            return []
        
        sections: List[Dict[str, Any]] = []
        section_map: Dict[str, Dict[str, Any]] = {}  # For tracking parent-child relationships
        
        for i, (level, title, page_number) in enumerate(toc):
            section_id = f"section_{i}"
            
            # Find parent based on TOC nesting level
            parent_id = None
            for j in range(i - 1, -1, -1):
                if sections[j]["level"] < level:
                    parent_id = sections[j]["id"]
                    break
            
            sections.append({
                "id": section_id,
                "title": title,
                "level": level,
                "page_number": page_number,
                "parent_id": parent_id
            })
            
            section_map[section_id] = sections[-1]
        
        return sections
    
    def _find_image_caption(self, text: str, figure_number: int) -> Optional[str]:
        """Try to find a caption for the image based on heuristics."""
        patterns: List[str] = [
            rf"Fig(?:ure)?\s*{figure_number}[\.:]\s*(.*?)(?:\n\n|\r\n\r\n|$)",
            rf"Figure\s*{figure_number}[\.:]\s*(.*?)(?:\n\n|\r\n\r\n|$)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                # Limit caption length
                caption = match.group(1).strip()
                return caption[:200] if len(caption) > 200 else caption
        
        return None
    
    def _extract_figure_references(self, text: str) -> List[str]:
        """Extract references to figures from text."""
        references: List[str] = []
        
        # Pattern matches "Fig. X", "Figure X", etc.
        pattern = r"Fig(?:ure)?\.?\s*(\d+)"
        matches = re.finditer(pattern, text, re.IGNORECASE)
        
        for match in matches:
            figure_num = match.group(1)
            references.append(f"Figure {figure_num}")
        
        return references
