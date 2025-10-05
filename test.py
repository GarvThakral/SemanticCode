"""
Sample Python Module
This is a comprehensive Python file demonstrating all major code structures.
"""

import os
import sys
from typing import List, Dict, Optional
from dataclasses import dataclass

# Module-level constants
API_KEY = "sk-test-123"
MAX_RETRIES = 3
DATABASE_URL = "postgresql://localhost/mydb"

# Module-level variable initialization
db_client = None
cache = {}


# Top-level function
def initialize_app(config: Dict) -> bool:
    """
    Initializes the application with given configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if initialization successful
    """
    global db_client
    db_client = connect_database(config.get('db_url'))
    return db_client is not None


def connect_database(url: str):
    """Connects to database"""
    print(f"Connecting to {url}")
    return {"connected": True}


# Function with nested function
def process_data(data: List[int]) -> List[int]:
    """
    Processes data with filtering and transformation.
    """
    
    def inner_filter(x):
        """Nested helper function"""
        return x > 0
    
    filtered = [x for x in data if inner_filter(x)]
    return [x * 2 for x in filtered]


# Decorator
def log_calls(func):
    """Decorator that logs function calls"""
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper


@log_calls
def decorated_function(x: int) -> int:
    """Function with decorator"""
    return x ** 2


# Dataclass
@dataclass
class Config:
    """Configuration data class"""
    api_key: str
    timeout: int = 30
    debug: bool = False


# Simple class
class DatabaseManager:
    """
    Manages database connections and queries.
    Handles connection pooling and query execution.
    """
    
    # Class variable
    connection_pool = []
    
    def __init__(self, url: str, pool_size: int = 5):
        """
        Initialize database manager.
        
        Args:
            url: Database connection URL
            pool_size: Size of connection pool
        """
        self.url = url
        self.pool_size = pool_size
        self._connection = None
    
    def connect(self) -> bool:
        """Establishes database connection"""
        self._connection = {"url": self.url, "active": True}
        return True
    
    def disconnect(self):
        """Closes database connection"""
        if self._connection:
            self._connection["active"] = False
            self._connection = None
    
    def query(self, sql: str) -> List[Dict]:
        """
        Executes SQL query.
        
        Args:
            sql: SQL query string
            
        Returns:
            List of result rows as dictionaries
        """
        if not self._connection:
            raise ConnectionError("Not connected to database")
        
        # Simulate query execution
        results = []
        return results
    
    @staticmethod
    def validate_sql(sql: str) -> bool:
        """Validates SQL syntax"""
        return len(sql) > 0 and "SELECT" in sql.upper()
    
    @classmethod
    def from_config(cls, config: Config):
        """Creates DatabaseManager from Config object"""
        return cls(url=DATABASE_URL, pool_size=10)
    
    @property
    def is_connected(self) -> bool:
        """Check if database is connected"""
        return self._connection is not None and self._connection.get("active", False)
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()


# Class with inheritance
class CachedDatabaseManager(DatabaseManager):
    """
    Extended DatabaseManager with caching capabilities.
    """
    
    def __init__(self, url: str, pool_size: int = 5, cache_ttl: int = 300):
        """Initialize with cache TTL"""
        super().__init__(url, pool_size)
        self.cache = {}
        self.cache_ttl = cache_ttl
    
    def query(self, sql: str) -> List[Dict]:
        """
        Executes SQL query with caching.
        Overrides parent query method.
        """
        if sql in self.cache:
            return self.cache[sql]
        
        results = super().query(sql)
        self.cache[sql] = results
        return results
    
    def clear_cache(self):
        """Clears query cache"""
        self.cache.clear()


# Abstract base class (using ABC)
from abc import ABC, abstractmethod

class DataProcessor(ABC):
    """Abstract base class for data processors"""
    
    @abstractmethod
    def process(self, data):
        """Process data - must be implemented by subclasses"""
        pass
    
    def validate(self, data) -> bool:
        """Validates input data"""
        return data is not None


class JSONProcessor(DataProcessor):
    """Processes JSON data"""
    
    def process(self, data):
        """Converts data to JSON format"""
        import json
        return json.dumps(data)


# Async function
async def fetch_data(url: str) -> Dict:
    """
    Asynchronously fetches data from URL.
    
    Args:
        url: Target URL
        
    Returns:
        Response data as dictionary
    """
    # Simulated async operation
    await asyncio.sleep(1)
    return {"url": url, "status": 200}


# Generator function
def generate_numbers(n: int):
    """
    Generator that yields numbers from 0 to n.
    
    Args:
        n: Upper limit
        
    Yields:
        Sequential integers
    """
    for i in range(n):
        yield i


# Lambda and one-liners
square = lambda x: x ** 2
is_even = lambda x: x % 2 == 0


# Exception class
class DatabaseError(Exception):
    """Custom exception for database errors"""
    
    def __init__(self, message: str, error_code: int = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


# Main execution block
if __name__ == "__main__":
    # Initialize
    config = {"db_url": DATABASE_URL}
    initialize_app(config)
    
    # Use database manager
    with DatabaseManager(DATABASE_URL) as db:
        results = db.query("SELECT * FROM users")
        print(f"Found {len(results)} results")
    
    # Process data
    data = [1, -2, 3, -4, 5]
    processed = process_data(data)
    print(f"Processed: {processed}")