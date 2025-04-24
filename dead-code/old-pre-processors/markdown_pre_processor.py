"""
Markdown pre-processor for the ingestion pipeline.

Processes Markdown files to extract:
- Headings and sections
- Code blocks
- Mermaid diagrams and their relationships
"""

from typing import Dict, Any, List, Optional, Tuple
import re
import os

from .base_pre_processor import BasePreProcessor


class MarkdownPreProcessor(BasePreProcessor):
    """Pre-processor for Markdown files, including Mermaid diagrams."""
    
    def __init__(self, extract_mermaid: bool = True):
        """
        Initialize the Markdown pre-processor.
        
        Args:
            extract_mermaid: Whether to extract and parse Mermaid diagrams
        """
        self.extract_mermaid = extract_mermaid
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a Markdown file, extracting:
        - Headings and sections
        - Code blocks
        - Mermaid diagrams
        
        Args:
            file_path: Path to Markdown file
            
        Returns:
            Structured document data
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Get relative path for document ID
        rel_path = os.path.basename(file_path)
        
        # Extract headings and sections
        sections = self._extract_sections(content)
        
        # Extract code blocks
        code_blocks: List[Dict[str, Any]] = self._extract_code_blocks(content)
        
        # Extract Mermaid diagrams if enabled
        mermaid_diagrams: List[Dict[str, Any]] = []
        mermaid_relationships: List[Dict[str, Any]] = []
        if self.extract_mermaid:
            mermaid_diagrams, mermaid_relationships = self._extract_mermaid(content, rel_path)
        
        # Extract title (first heading or section title)
        title = None
        for section in sections:
            if section.get('title'):
                title = section['title']
                break
        # Extract references (very basic: look for links in content)
        references: List[Dict[str, Any]] = []
        link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        for match in re.finditer(link_pattern, content):
            references.append({'label': match.group(1), 'target': match.group(2)})

        document = {
            'path': file_path,
            'id': rel_path,
            'type': 'markdown',
            'content': content,
            'sections': sections,
            'code_blocks': code_blocks,
            'mermaid_diagrams': mermaid_diagrams,
            'relationships': mermaid_relationships,
            'title': title,
            'references': references,
        }
        return document
    
    def _extract_sections(self, content: str) -> List[Dict[str, Any]]:
        """Extract headings and sections from Markdown content."""
        # Regular expression to find headings
        heading_pattern = r'^(#{1,6})\s+(.+?)(?:\s+#+)?$'
        
        sections: List[Dict[str, Any]] = []
        current_position = 0
        
        for match in re.finditer(heading_pattern, content, re.MULTILINE):
            heading_level = len(match.group(1))
            heading_text = match.group(2).strip()
            start_pos = match.start()
            
            # Add a section for content before this heading if there is any
            if start_pos > current_position:
                section_content = content[current_position:start_pos].strip()
                if section_content:
                    sections.append({
                        'level': 0,  # Level 0 for non-heading content
                        'title': None,
                        'content': section_content,
                        'position': current_position,
                    })
            
            # Update current position to after this heading
            current_position = match.end()
            
            # Find end position - next heading or end of document
            next_match = re.search(heading_pattern, content[current_position:], re.MULTILINE)
            end_pos = current_position + next_match.start() if next_match else len(content)
            
            # Extract section content
            section_content = content[current_position:end_pos].strip()
            
            # Add this section
            sections.append({
                'level': heading_level,
                'title': heading_text,
                'content': section_content,
                'position': current_position,
            })
            
            # Update current position
            current_position = end_pos
        
        # Add any remaining content
        if current_position < len(content):
            section_content = content[current_position:].strip()
            if section_content:
                sections.append({
                    'level': 0,
                    'title': None,
                    'content': section_content,
                    'position': current_position,
                })
        
        return sections
    
    def _extract_code_blocks(self, content: str) -> List[Dict[str, Any]]:
        """Extract code blocks from Markdown content."""
        # Regular expression to find code blocks
        code_block_pattern = r'```([a-zA-Z0-9]*)\n(.*?)```'
        
        code_blocks: List[Dict[str, Any]] = []
        
        for match in re.finditer(code_block_pattern, content, re.DOTALL):
            language = match.group(1) or 'text'
            code_content = match.group(2)
            

                
            code_blocks.append({
                'language': language,
                'content': code_content,
                'position': match.start(),
            })
        
        return code_blocks
    
    def _extract_mermaid(self, content: str, document_id: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Extract Mermaid diagram blocks and parse them.
        
        Args:
            content: Markdown content
            document_id: Document ID for relationship building
            
        Returns:
            Tuple of (diagrams, relationships)
        """
        # Regular expression to find Mermaid blocks
        mermaid_pattern = r'```mermaid\n(.*?)```'
        
        diagrams: List[Dict[str, Any]] = []
        relationships: List[Dict[str, Any]] = []
        
        for idx, match in enumerate(re.finditer(mermaid_pattern, content, re.DOTALL)):
            diagram_text = match.group(1)
            diagram_id = f"{document_id}::diagram_{idx}"
            
            # Parse the diagram
            diagram_data, diagram_rels = self._parse_mermaid_diagram(diagram_text, diagram_id)
            
            # Add diagram info
            diagrams.append({
                'id': diagram_id,
                'content': diagram_text,
                'position': match.start(),
                'nodes': diagram_data.get('nodes', []),
                'type': diagram_data.get('type', 'unknown'),
            })
            
            # Add relationships
            relationships.extend(diagram_rels)
        
        return diagrams, relationships
    
    def _parse_mermaid_diagram(self, diagram_text: str, diagram_id: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Parse a Mermaid diagram, extracting nodes and relationships.
        
        Args:
            diagram_text: The Mermaid diagram text
            diagram_id: ID of the diagram for relationships
            
        Returns:
            Tuple of (diagram_data, relationships)
        """
        diagram_type = self._detect_diagram_type(diagram_text)
        nodes: List[Dict[str, Any]] = []
        relationships: List[Dict[str, Any]] = []
        
        if diagram_type == 'flowchart':
            nodes, relationships = self._parse_flowchart(diagram_text, diagram_id)
        elif diagram_type == 'sequence':
            nodes, relationships = self._parse_sequence_diagram(diagram_text, diagram_id)
        elif diagram_type == 'class':
            nodes, relationships = self._parse_class_diagram(diagram_text, diagram_id)
        elif diagram_type == 'er':
            nodes, relationships = self._parse_er_diagram(diagram_text, diagram_id)
        
        # Create diagram data
        diagram_data = {
            'type': diagram_type,
            'nodes': nodes,
        }
        
        return diagram_data, relationships
    
    def _detect_diagram_type(self, diagram_text: str) -> str:
        """Detect the type of Mermaid diagram."""
        first_line = diagram_text.strip().split('\n')[0].lower()
        
        if 'flowchart' in first_line or 'graph' in first_line:
            return 'flowchart'
        elif 'sequencediagram' in first_line.replace(' ', ''):
            return 'sequence'
        elif 'classDiagram' in first_line.replace(' ', ''):
            return 'class'
        elif 'erDiagram' in first_line.replace(' ', ''):
            return 'er'
        return 'unknown'
    
    def _parse_flowchart(self, diagram_text: str, diagram_id: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Parse a Mermaid flowchart diagram.
        
        Args:
            diagram_text: The Mermaid diagram text
            diagram_id: ID of the diagram for relationships
            
        Returns:
            Tuple of (nodes, relationships)
        """
        # Regular expressions for nodes and connections
        node_pattern = r'([A-Za-z0-9_-]+)(?:\["([^"]+)"\]|\[([^\]]+)\]|(?:\("([^"]+)"\)|(?:\(([^)]+)\)))|\{([^}]+)\})?'
        edge_pattern = r'([A-Za-z0-9_-]+)\s*(-->|---|-.-|--o|--x|-.->|===>|---|~~~)\s*([A-Za-z0-9_-]+)(?:\s*\|([^|]+)\|)?'
        
        nodes: List[Dict[str, Any]] = []
        relationships: List[Dict[str, Any]] = []
        node_ids = set()
        
        # Find nodes with explicit content
        for match in re.finditer(node_pattern, diagram_text, re.MULTILINE):
            node_id = match.group(1)
            # Find the first non-None group for label
            label = next((g for g in match.groups()[1:] if g is not None), node_id)
            
            if node_id not in node_ids:
                nodes.append({
                    'id': node_id,
                    'label': label,
                    'type': 'flowchart_node',
                })
                node_ids.add(node_id)
        
        # Find edges
        for match in re.finditer(edge_pattern, diagram_text, re.MULTILINE):
            source = match.group(1)
            target = match.group(3)
            edge_type = match.group(2)
            label = match.group(4) if match.group(4) else ''
            
            # Ensure nodes exist (might be implicit nodes)
            for node_id in [source, target]:
                if node_id not in node_ids:
                    nodes.append({
                        'id': node_id,
                        'label': node_id,
                        'type': 'flowchart_node',
                    })
                    node_ids.add(node_id)
            
            # Map edge types to relationship types
            rel_type = 'CONNECTS_TO'
            if '-->' in edge_type or '==>' in edge_type or '-.->':
                rel_type = 'LEADS_TO'
            elif '--o' in edge_type:
                rel_type = 'REFERENCES'
            elif '--x' in edge_type:
                rel_type = 'TERMINATES_AT'
            
            # Add relationship
            relationships.append({
                'from': f"{diagram_id}::{source}",
                'to': f"{diagram_id}::{target}",
                'type': rel_type,
                'weight': 0.6,
                'label': label,
            })
        
        return nodes, relationships
    
    def _parse_sequence_diagram(self, diagram_text: str, diagram_id: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Parse a sequence diagram."""
        # Regular expressions for participants and messages
        participant_pattern = r'participant\s+([A-Za-z0-9_-]+)(?:\s+as\s+([A-Za-z0-9_-]+))?'
        actor_pattern = r'actor\s+([A-Za-z0-9_-]+)(?:\s+as\s+([A-Za-z0-9_-]+))?'
        message_pattern = r'([A-Za-z0-9_-]+)\s*(->>|-->|--x|-[xX])\s*([A-Za-z0-9_-]+)\s*:\s*(.+)'
        
        nodes: List[Dict[str, Any]] = []
        relationships: List[Dict[str, Any]] = []
        participant_ids = set()
        
        # Find participants
        for pattern in [participant_pattern, actor_pattern]:
            for match in re.finditer(pattern, diagram_text, re.MULTILINE):
                participant_name = match.group(1)
                participant_alias = match.group(2) if match.group(2) else participant_name
                
                if participant_alias not in participant_ids:
                    nodes.append({
                        'id': participant_alias,
                        'label': participant_name,
                        'type': 'sequence_participant',
                    })
                    participant_ids.add(participant_alias)
        
        # Find messages
        for match in re.finditer(message_pattern, diagram_text, re.MULTILINE):
            source = match.group(1)
            target = match.group(3)
            message_type = match.group(2)
            message = match.group(4).strip()
            
            # Ensure participants exist
            for participant in [source, target]:
                if participant not in participant_ids:
                    nodes.append({
                        'id': participant,
                        'label': participant,
                        'type': 'sequence_participant',
                    })
                    participant_ids.add(participant)
            
            # Determine relationship type
            rel_type = 'SENDS_TO'
            if '--x' in message_type or '-x' in message_type or '-X' in message_type:
                rel_type = 'RESPONDS_TO'
            
            # Add relationship
            relationships.append({
                'from': f"{diagram_id}::{source}",
                'to': f"{diagram_id}::{target}",
                'type': rel_type,
                'weight': 0.7,
                'label': message,
            })
        
        return nodes, relationships
    
    def _parse_class_diagram(self, diagram_text: str, diagram_id: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Parse a class diagram."""
        # Regular expressions for classes and relationships
        class_pattern = r'class\s+([A-Za-z0-9_-]+)\s*(?:{([^}]*)})?\s*'
        relationship_pattern = r'([A-Za-z0-9_-]+)\s*(<[|*.]+|-+|<\|--|<\.\.|\.\.|<\|\.\.)-+(>|[|*.]+>)?\s*([A-Za-z0-9_-]+)\s*:\s*(.+)?'
        
        nodes: List[Dict[str, Any]] = []
        relationships: List[Dict[str, Any]] = []
        class_ids = set()
        
        # Find classes
        for match in re.finditer(class_pattern, diagram_text, re.MULTILINE):
            class_name = match.group(1)
            class_body = match.group(2) if match.group(2) else ''
            
            if class_name not in class_ids:
                nodes.append({
                    'id': class_name,
                    'label': class_name,
                    'type': 'class',
                    'body': class_body,
                })
                class_ids.add(class_name)
        
        # Find relationships
        for match in re.finditer(relationship_pattern, diagram_text, re.MULTILINE):
            source = match.group(1)
            target = match.group(4)
            rel_symbol = match.group(2) if match.group(2) else ''
            if match.group(3):
                rel_symbol += match.group(3)
            label = match.group(5) if match.group(5) else ''
            
            # Ensure classes exist
            for class_name in [source, target]:
                if class_name not in class_ids:
                    nodes.append({
                        'id': class_name,
                        'label': class_name,
                        'type': 'class',
                    })
                    class_ids.add(class_name)
            
            # Map relationship symbols to types
            rel_type = 'RELATED_TO'
            weight = 0.5
            
            if '<|--' in rel_symbol or '<|..' in rel_symbol:
                rel_type = 'EXTENDS'
                weight = 0.7
            elif '<--' in rel_symbol or '<..' in rel_symbol:
                rel_type = 'USES'
                weight = 0.6
            elif '*--' in rel_symbol or '*..' in rel_symbol:
                rel_type = 'CONTAINS'
                weight = 0.9
            elif 'o--' in rel_symbol or 'o..' in rel_symbol:
                rel_type = 'REFERENCES'
                weight = 0.5
            
            # Add relationship
            relationships.append({
                'from': f"{diagram_id}::{source}",
                'to': f"{diagram_id}::{target}",
                'type': rel_type,
                'weight': weight,
                'label': label,
            })
        
        return nodes, relationships
    
    def _parse_er_diagram(self, diagram_text: str, diagram_id: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Parse an ER diagram."""
        # Regular expressions for entities and relationships
        entity_pattern = r'([A-Za-z0-9_-]+)\s*{([^}]*)}'
        relationship_pattern = r'([A-Za-z0-9_-]+)\s+([|o*])\-\-([|o*])\s+([A-Za-z0-9_-]+)\s*:\s*("([^"]+)"|([^\s"]+))'
        
        nodes: List[Dict[str, Any]] = []
        relationships: List[Dict[str, Any]] = []
        entity_ids = set()
        
        # Find entities
        for match in re.finditer(entity_pattern, diagram_text, re.MULTILINE):
            entity_name = match.group(1)
            entity_attrs = match.group(2) if match.group(2) else ''
            
            if entity_name not in entity_ids:
                nodes.append({
                    'id': entity_name,
                    'label': entity_name,
                    'type': 'er_entity',
                    'attributes': entity_attrs,
                })
                entity_ids.add(entity_name)
        
        # Find relationships
        for match in re.finditer(relationship_pattern, diagram_text, re.MULTILINE):
            source = match.group(1)
            source_card = match.group(2)
            target_card = match.group(3)
            target = match.group(4)
            relationship_name = match.group(6) if match.group(6) else match.group(7)
            
            # Ensure entities exist
            for entity_name in [source, target]:
                if entity_name not in entity_ids:
                    nodes.append({
                        'id': entity_name,
                        'label': entity_name,
                        'type': 'er_entity',
                    })
                    entity_ids.add(entity_name)
            
            # Add relationship
            relationships.append({
                'from': f"{diagram_id}::{source}",
                'to': f"{diagram_id}::{target}",
                'type': 'RELATES_TO',
                'weight': 0.8,
                'label': relationship_name,
                'source_cardinality': source_card,
                'target_cardinality': target_card,
            })
        
        return nodes, relationships
