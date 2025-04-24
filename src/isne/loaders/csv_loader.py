"""
CSV loader for the ISNE pipeline.

This module provides a loader for reading documents from CSV files,
supporting customizable column mapping and relationship extraction.
"""

import csv
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Set, Iterator, Tuple, cast
import logging
from datetime import datetime
import io

from src.isne.types.models import IngestDocument, DocumentRelation, RelationType
from src.isne.loaders.base_loader import BaseLoader, LoaderConfig, LoaderResult

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CSVLoader(BaseLoader):
    """
    Loader for documents from CSV files.
    
    This loader extracts documents from CSV files with customizable
    column mapping and optional relationship extraction.
    """
    
    def __init__(
        self, 
        config: Optional[LoaderConfig] = None,
        document_type: str = "csv",
        id_column: Optional[str] = "id",
        content_column: str = "content",
        title_column: Optional[str] = "title",
        source_column: Optional[str] = None,
        delimiter: str = ",",
        quotechar: str = '"',
        metadata_columns: Optional[List[str]] = None,
        relation_source_column: Optional[str] = None,
        relation_target_column: Optional[str] = None,
        relation_type_column: Optional[str] = None
    ) -> None:
        """
        Initialize the CSV loader.
        
        Args:
            config: Loader configuration
            document_type: Type to assign to loaded documents
            id_column: CSV column containing document IDs (optional)
            content_column: CSV column containing document content
            title_column: CSV column containing document titles (optional)
            source_column: CSV column containing document sources (optional)
            delimiter: CSV delimiter character
            quotechar: CSV quote character
            metadata_columns: List of CSV columns to include as metadata (optional)
            relation_source_column: CSV column for relationship source IDs (optional)
            relation_target_column: CSV column for relationship target IDs (optional)
            relation_type_column: CSV column for relationship types (optional)
        """
        super().__init__(config)
        
        self.document_type = document_type
        self.id_column = id_column
        self.content_column = content_column
        self.title_column = title_column
        self.source_column = source_column
        self.delimiter = delimiter
        self.quotechar = quotechar
        self.metadata_columns = metadata_columns or []
        self.relation_source_column = relation_source_column
        self.relation_target_column = relation_target_column
        self.relation_type_column = relation_type_column
    
    def load(self, source: Union[str, Path]) -> LoaderResult:
        """
        Load documents from a CSV file.
        
        Args:
            source: Path to the CSV file to load documents from
            
        Returns:
            LoaderResult containing loaded documents and relationships
        """
        source_path = Path(source)
        
        if not source_path.exists():
            raise ValueError(f"Source file does not exist: {source}")
        
        if not source_path.is_file():
            raise ValueError(f"Source is not a file: {source}")
        
        logger.info(f"Loading documents from CSV file: {source_path}")
        
        # Read CSV file
        with open(source_path, 'r', encoding=self.config.encoding) as f:
            # Detect if it's actually a TSV file
            if self.delimiter == "," and "\t" in f.readline() and "," not in f.readline():
                logger.info("File appears to be tab-delimited, switching to TSV mode")
                self.delimiter = "\t"
            
            # Reset file position
            f.seek(0)
            
            # Parse CSV
            csv_reader = csv.DictReader(
                f, 
                delimiter=self.delimiter,
                quotechar=self.quotechar
            )
            
            # Validate required columns
            required_columns = [self.content_column]
            if self.id_column:
                required_columns.append(self.id_column)
            if self.title_column:
                required_columns.append(self.title_column)
            if self.source_column:
                required_columns.append(self.source_column)
            
            # Check for relationship column requirements
            relation_mode = False
            if self.relation_source_column and self.relation_target_column:
                relation_mode = True
                required_columns.extend([self.relation_source_column, self.relation_target_column])
                if self.relation_type_column:
                    required_columns.append(self.relation_type_column)
            
            # Verify all required columns exist
            if csv_reader.fieldnames:
                missing_columns = [col for col in required_columns if col not in csv_reader.fieldnames]
                if missing_columns:
                    raise ValueError(f"Missing required columns in CSV: {', '.join(missing_columns)}")
            else:
                raise ValueError("CSV file has no headers")
            
            # Process each row
            documents: List[IngestDocument] = []
            relations: List[DocumentRelation] = []
            errors: List[Dict[str, Any]] = []
            
            for i, row in enumerate(csv_reader):
                try:
                    # If in relation mode, extract relationships
                    if relation_mode:
                        relation = self._process_relation_row(row, i)
                        if relation:
                            relations.append(relation)
                    # Otherwise extract documents
                    else:
                        document = self._process_document_row(row, source_path, i)
                        documents.append(document)
                except Exception as e:
                    errors.append({
                        "row": i + 2,  # +2 for header row and 0-indexing
                        "error": str(e),
                        "type": type(e).__name__
                    })
                    logger.error(f"Error processing row {i + 2}: {e}")
            
            # Create dataset
            dataset_name = f"csv_{source_path.stem}"
            dataset = self.create_dataset(
                name=dataset_name,
                documents=documents,
                relations=relations,
                description=f"Documents from CSV file: {source_path}",
                metadata={
                    "source_path": str(source_path),
                    "document_count": len(documents),
                    "relationship_count": len(relations),
                    "error_count": len(errors)
                }
            )
            
            return LoaderResult(
                documents=documents,
                relations=relations,
                dataset=dataset,
                errors=errors,
                metadata={
                    "source_path": str(source_path),
                    "document_count": len(documents),
                    "relation_count": len(relations),
                    "error_count": len(errors)
                }
            )
    
    def _process_document_row(self, row: Dict[str, str], source_path: Path, row_index: int) -> IngestDocument:
        """
        Process a single document from a CSV row.
        
        Args:
            row: CSV row data
            source_path: Path to the source CSV file
            row_index: Index of the row in the CSV file
            
        Returns:
            Processed IngestDocument
        """
        # Extract content (required)
        content = row[self.content_column]
        if not content.strip():
            content = f"Empty content in row {row_index + 2}"
        
        # Extract document ID (generate if not present)
        doc_id = row.get(self.id_column, "") if self.id_column else ""
        if not doc_id:
            doc_id = str(uuid.uuid4())
        
        # Extract title
        title = None
        if self.title_column and self.title_column in row:
            title = row[self.title_column]
        
        # Extract source
        source = None
        if self.source_column and self.source_column in row:
            source = row[self.source_column]
        
        if not source:
            source = f"{source_path.name}:{row_index + 2}"
        
        # Extract metadata
        metadata: Dict[str, Any] = {}
        
        # Include specific metadata columns
        for col in self.metadata_columns:
            if col in row:
                metadata[col] = row[col]
        
        # Optionally include all other columns as metadata
        if not self.metadata_columns:
            exclude_cols = {self.id_column, self.content_column, 
                           self.title_column, self.source_column}
            
            for col, value in row.items():
                if col not in exclude_cols and col is not None:
                    metadata[col] = value
        
        # Create document
        return IngestDocument(
            id=doc_id,
            content=content,
            source=source,
            document_type=self.document_type,
            title=title,
            metadata=metadata
        )
    
    def _process_relation_row(self, row: Dict[str, str], row_index: int) -> Optional[DocumentRelation]:
        """
        Process a single relationship from a CSV row.
        
        Args:
            row: CSV row data
            row_index: Index of the row in the CSV file
            
        Returns:
            Processed DocumentRelation or None if invalid
        """
        # Extract source and target IDs (required)
        source_id = row.get(self.relation_source_column, "") if self.relation_source_column else ""
        target_id = row.get(self.relation_target_column, "") if self.relation_target_column else ""
        
        # Skip if missing source or target
        if not source_id or not target_id:
            logger.warning(f"Skipping relation in row {row_index + 2}: missing source or target ID")
            return None
        
        # Extract relation type
        relation_type_str = ""
        if self.relation_type_column and self.relation_type_column in row:
            relation_type_str = row[self.relation_type_column]
        
        # Convert relation type to enum
        try:
            if relation_type_str:
                relation_type = RelationType(relation_type_str.lower())
            else:
                relation_type = RelationType.RELATED_TO
        except ValueError:
            relation_type = RelationType.CUSTOM
        
        # Extract weight if present
        weight = 1.0
        if "weight" in row:
            try:
                weight = float(row["weight"])
            except ValueError:
                pass
        
        # Extract bidirectional flag if present
        bidirectional = False
        if "bidirectional" in row:
            bidirectional = row["bidirectional"].lower() in ("true", "yes", "1")
        
        # Extract metadata
        metadata: Dict[str, Any] = {}
        exclude_cols = {
            self.relation_source_column, 
            self.relation_target_column, 
            self.relation_type_column,
            "weight", "bidirectional"
        }
        
        for col, value in row.items():
            if col not in exclude_cols and col is not None:
                metadata[col] = value
        
        # Create relationship
        return DocumentRelation(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            weight=weight,
            bidirectional=bidirectional,
            metadata=metadata
        )
