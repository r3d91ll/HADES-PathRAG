"""
Simplified authentication for the HADES MCP server (POC Implementation).

This module implements a minimal authentication approach for the 
Model Context Protocol (MCP) server, designed for local proof-of-concept testing.

The simplified implementation uses an in-memory store for API keys,
eliminating the need for PostgreSQL or SQLite databases.
"""
import os
import time
import uuid
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any

from fastapi import Depends, HTTPException, Request, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from src.utils.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# API key header
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


class APIKey(BaseModel):
    """API key model."""
    key_id: str
    name: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    is_active: bool = True


class SimpleAuthManager:
    """
    Simplified authentication manager for POC implementation.
    
    This class implements a minimal in-memory authentication system
    for the HADES MCP server, designed for local development and testing.
    It eliminates the need for external database dependencies.
    """
    
    def __init__(self):
        """Initialize the auth manager."""
        # In-memory storage for API keys and rate limits
        self.api_keys = {}
        self.rate_limits = {}
        
        # Create a default API key for development
        self._create_default_key()
        
    def _create_default_key(self):
        """Create a default API key for development."""
        default_key_id = "dev-key"
        default_api_key = "dev-api-key-for-local-testing-only"
        
        self.api_keys[default_key_id] = {
            "key_id": default_key_id,
            "api_key": default_api_key,
            "name": "Default Development Key",
            "created_at": datetime.now(),
            "expires_at": None,
            "is_active": True
        }
        
        logger.info(f"Created default development API key: {default_api_key}")
        logger.info("This key is for local development only and should not be used in production.")
    
    def create_api_key(self, name: str, expiry_days: Optional[int] = None) -> Tuple[str, str]:
        """
        Create a new API key.
        
        Args:
            name: Name or purpose of the key
            expiry_days: Number of days until the key expires (None for no expiration)
            
        Returns:
            Tuple of key_id and the actual API key
        """
        key_id = str(uuid.uuid4())
        api_key = str(uuid.uuid4())
        
        created_at = datetime.now()
        expires_at = None
        if expiry_days is not None:
            expires_at = created_at + timedelta(days=expiry_days)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if self.db_type == "sqlite":
                # Format datetime as ISO string for SQLite
                created_at_str = created_at.isoformat()
                expires_at_str = expires_at.isoformat() if expires_at else None
                
                cursor.execute(
                    """
                    INSERT INTO api_keys (key_id, key_hash, name, created_at, expires_at, is_active)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (key_id, key_hash, name, created_at_str, expires_at_str, True)
                )
            else:  # postgresql
                cursor.execute(
                    """
                    INSERT INTO api_keys (key_id, key_hash, name, created_at, expires_at, is_active)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (key_id, key_hash, name, created_at, expires_at, True)
                )
            
            conn.commit()
        
        return key_id, api_key
    
    def validate_api_key(self, api_key: str) -> Optional[APIKey]:
        """
        Validate an API key.
        
        Args:
            api_key: The API key to validate
            
        Returns:
            APIKey object if valid, None otherwise
        """
        if not api_key:
            return None
        
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if self.db_type == "sqlite":
                cursor.execute(
                    """
                    SELECT key_id, name, created_at, expires_at, is_active 
                    FROM api_keys 
                    WHERE key_hash = ?
                    """,
                    (key_hash,)
                )
            else:  # postgresql
                cursor.execute(
                    """
                    SELECT key_id, name, created_at, expires_at, is_active 
                    FROM api_keys 
                    WHERE key_hash = %s
                    """,
                    (key_hash,)
                )
            
            row = cursor.fetchone()
            
            if not row:
                return None
            
            # Check if the key is active
            if not row["is_active"]:
                return None
            
            # Check if the key has expired
            now = datetime.now()
            expires_at = None
            
            if self.db_type == "sqlite" and row["expires_at"]:
                # Convert ISO string to datetime for SQLite
                expires_at = datetime.fromisoformat(row["expires_at"])
                if expires_at < now:
                    return None
            elif self.db_type == "postgresql" and row["expires_at"]:
                # PostgreSQL returns datetime objects directly
                expires_at = row["expires_at"]
                if expires_at < now:
                    return None
            
            # Create APIKey object
            created_at = row["created_at"]
            if self.db_type == "sqlite":
                # Convert ISO string to datetime for SQLite
                created_at = datetime.fromisoformat(created_at)
                if row["expires_at"]:
                    expires_at = datetime.fromisoformat(row["expires_at"])
            
            return APIKey(
                key_id=row["key_id"],
                name=row["name"],
                created_at=created_at,
                expires_at=expires_at,
                is_active=bool(row["is_active"])
            )
    
    def check_rate_limit(self, api_key: str, rpm_limit: int = None) -> bool:
        """
        Check if a key has exceeded its rate limit.
        
        Args:
            api_key: The API key to check
            rpm_limit: Requests per minute limit (defaults to config value)
            
        Returns:
            True if within limits, False if exceeded
        """
        if rpm_limit is None:
            rpm_limit = config.mcp.auth.rate_limit_rpm
        if not api_key:
            return False
        
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        now = datetime.now()
        window_start = now - timedelta(minutes=1)
        expires_at = now + timedelta(minutes=5)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Clean up expired rate limits
            if self.db_type == "sqlite":
                cursor.execute(
                    "DELETE FROM rate_limits WHERE expires_at < ?",
                    (now.isoformat(),)
                )
            else:  # postgresql
                cursor.execute(
                    "DELETE FROM rate_limits WHERE expires_at < %s",
                    (now,)
                )
            
            # Get current request count in the time window
            if self.db_type == "sqlite":
                cursor.execute(
                    """
                    SELECT SUM(requests) as total_requests
                    FROM rate_limits
                    WHERE key_id = ? AND window_start >= ?
                    """,
                    (key_hash, window_start.isoformat())
                )
            else:  # postgresql
                cursor.execute(
                    """
                    SELECT SUM(requests) as total_requests
                    FROM rate_limits
                    WHERE key_id = %s AND window_start >= %s
                    """,
                    (key_hash, window_start)
                )
            
            row = cursor.fetchone()
            total_requests = row["total_requests"] if row and row["total_requests"] else 0
            
            # If we're already over the limit, deny the request
            if total_requests >= rpm_limit:
                return False
            
            # Record this request
            if self.db_type == "sqlite":
                cursor.execute(
                    """
                    INSERT INTO rate_limits (key_id, requests, window_start, expires_at)
                    VALUES (?, 1, ?, ?)
                    """,
                    (key_hash, now.isoformat(), expires_at.isoformat())
                )
            else:  # postgresql
                cursor.execute(
                    """
                    INSERT INTO rate_limits (key_id, requests, window_start, expires_at)
                    VALUES (%s, 1, %s, %s)
                    """,
                    (key_hash, now, expires_at)
                )
            
            conn.commit()
            
            return True


# Global auth manager instance
auth_manager = SimpleAuthManager()


async def get_api_key(
    api_key: str = Security(API_KEY_HEADER),
) -> Optional[APIKey]:
    """
    Dependency for validating API keys.
    
    Args:
        api_key: The API key from the request header
        
    Returns:
        APIKey if valid, None otherwise
    """
    if not config.mcp.auth_enabled:
        # If auth is not enabled, return a dummy key
        return APIKey(
            key_id="dummy",
            name="anonymous",
            created_at=datetime.now(),
        )
    
    if not api_key:
        return None
    
    return auth_db.validate_api_key(api_key)


async def get_current_key(
    api_key: Optional[APIKey] = Depends(get_api_key),
) -> APIKey:
    """
    Dependency for requiring a valid API key.
    
    Args:
        api_key: The validated API key
        
    Returns:
        APIKey if valid
        
    Raises:
        HTTPException if invalid
    """
    if not config.mcp.auth_enabled:
        # If auth is not enabled, return a dummy key
        return APIKey(
            key_id="dummy",
            name="anonymous",
            created_at=datetime.now(),
        )
    
    if api_key is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "APIKey"},
        )
    
    return api_key


async def check_rate_limit(
    request: Request,
    api_key: APIKey = Depends(get_current_key),
) -> None:
    """
    Dependency for checking rate limits.
    
    Args:
        request: The FastAPI request
        api_key: The validated API key
        
    Raises:
        HTTPException if rate limit exceeded
    """
    if not config.mcp.auth_enabled:
        return
    
    raw_key = request.headers.get("X-API-Key")
    if not raw_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key",
            headers={"WWW-Authenticate": "APIKey"},
        )
    
    if not auth_db.check_rate_limit(raw_key):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
        )
