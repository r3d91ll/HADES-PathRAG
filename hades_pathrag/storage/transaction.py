"""
Transaction support for ArangoDB storage.

This module provides classes for atomic operations across multiple
ArangoDB collections using transactions.
"""
from typing import Dict, List, Optional, Any, Union, Callable, TypeVar, cast
import logging
from contextlib import contextmanager

from arango import ArangoClient
from arango.database import Database
from arango.collection import Collection
from arango.graph import Graph as ArangoGraph
from arango.exceptions import (
    TransactionInitError,
    TransactionCommitError,
    TransactionAbortError,
)

from .interfaces import StorageTransaction
from .arango import ArangoDBConnection

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ArangoTransaction(StorageTransaction):
    """
    Transaction manager for ArangoDB.
    
    This class provides transaction support for atomic operations
    across multiple ArangoDB collections.
    """
    
    def __init__(
        self,
        connection: ArangoDBConnection,
        collections_read: Optional[List[str]] = None,
        collections_write: Optional[List[str]] = None,
        max_retries: int = 3,
    ):
        """
        Initialize transaction manager.
        
        Args:
            connection: ArangoDB connection
            collections_read: Collections to read from
            collections_write: Collections to write to
            max_retries: Maximum number of retries for transaction
        """
        self.connection = connection
        self.collections_read = collections_read or []
        self.collections_write = collections_write or []
        self.max_retries = max_retries
        self._tx_id = None
        self._active = False
    
    def begin(self) -> bool:
        """
        Begin a transaction.
        
        Returns:
            True if transaction started successfully, False otherwise
        """
        if self._active:
            logger.warning("Transaction already active")
            return False
        
        db = self.connection.connect()
        
        try:
            # Start transaction
            # Using type ignore since the ArangoDB types don't have proper stubs
            tx = db.begin_transaction(  # type: ignore
                read=self.collections_read,
                write=self.collections_write,
                exclusive=self.collections_write,
                sync=True
            )
            self._tx_id = tx.id
            self._active = True
            return True
        except TransactionInitError as e:
            logger.error(f"Error beginning transaction: {e}")
            return False
    
    def commit(self) -> bool:
        """
        Commit the transaction.
        
        Returns:
            True if committed successfully, False otherwise
        """
        # Validate transaction state
        valid_transaction = self._active and self._tx_id is not None
        if not valid_transaction:
            logger.warning("No active transaction to commit")
            return False
            
        # Get database connection and commit the transaction
        try:
            db = self.connection.connect()
            db.commit_transaction(self._tx_id)  # type: ignore
            self._active = False
            return True
        except TransactionCommitError as e:
            logger.error(f"Error committing transaction: {e}")
            return False
    
    def abort(self) -> bool:
        """
        Abort the transaction.
        
        Returns:
            True if aborted successfully, False otherwise
        """
        # Validate transaction state
        valid_transaction = self._active and self._tx_id is not None
        if not valid_transaction:
            logger.warning("No active transaction to abort")
            return False
            
        # Get database connection and abort the transaction
        try:
            db = self.connection.connect()
            db.abort_transaction(self._tx_id)  # type: ignore
            self._active = False
            return True
        except TransactionAbortError as e:
            logger.error(f"Error aborting transaction: {e}")
            return False
    
    def is_active(self) -> bool:
        """
        Check if the transaction is active.
        
        Returns:
            True if transaction is active, False otherwise
        """
        return self._active
    
    def __enter__(self) -> 'StorageTransaction':
        """
        Context manager entry.
        
        Returns:
            Transaction instance
        """
        success = self.begin()
        if not success:
            raise RuntimeError("Failed to begin transaction")
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """
        Context manager exit.
        
        Args:
            exc_type: Exception type if an exception occurred, None otherwise
            exc_val: Exception value if an exception occurred, None otherwise
            exc_tb: Exception traceback if an exception occurred, None otherwise
        """
        if exc_type is not None:
            # An exception occurred, roll back the transaction
            logger.error(f"Error in transaction: {exc_val}")
            self.rollback()
        else:
            # No exception, commit the transaction
            success = self.commit()
            if not success:
                raise RuntimeError("Failed to commit transaction")
    
    def rollback(self) -> None:
        """
        Roll back the transaction.
        """
        self.abort()
    
    @contextmanager
    def transaction_context(self) -> Any:
        """
        Context manager for transactions.
        
        Usage:
            with transaction.transaction_context():
                # Do operations
        """
        success = self.begin()
        if not success:
            raise RuntimeError("Failed to begin transaction")
        
        try:
            yield
            success = self.commit()
            if not success:
                raise RuntimeError("Failed to commit transaction")
        except Exception as e:
            logger.error(f"Error in transaction: {e}")
            self.abort()
            raise
    
    def execute_in_transaction(self, func: Callable[[], T]) -> Optional[T]:
        """
        Execute a function within a transaction.
        
        Args:
            func: Function to execute
            
        Returns:
            Result of the function if successful, None otherwise
        """
        with self.transaction_context():
            return func()


def create_transaction(
    connection: ArangoDBConnection,
    collections_read: Optional[List[str]] = None,
    collections_write: Optional[List[str]] = None,
) -> ArangoTransaction:
    """
    Create a transaction manager.
    
    Args:
        connection: ArangoDB connection
        collections_read: Collections to read from
        collections_write: Collections to write to
        
    Returns:
        Transaction manager
    """
    return ArangoTransaction(
        connection=connection,
        collections_read=collections_read,
        collections_write=collections_write
    )
