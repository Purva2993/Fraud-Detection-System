"""Configuration management for the fraud detection system."""

import os
from pathlib import Path
from dataclasses import dataclass


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    sqlite_path: str = "data/fraud_detection.db"


@dataclass
class ModelConfig:
    """Model configuration settings."""
    model_type: str = "xgboost"
    test_size: float = 0.2
    random_state: int = 42
    model_path: str = "models"


@dataclass
class StreamingConfig:
    """Streaming configuration settings."""
    batch_size: int = 100
    processing_interval: int = 5  # seconds


@dataclass
class APIConfig:
    """API configuration settings."""
    host: str = "localhost"
    port: int = 8000
    log_level: str = "INFO"


class Config:
    """Main configuration class."""
    
    def __init__(self):
        # Initialize with default configurations
        self.database = DatabaseConfig()
        self.model = ModelConfig()
        self.streaming = StreamingConfig()
        self.api = APIConfig()
        
        # Create necessary directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary project directories."""
        directories = [
            "data/raw",
            "data/processed", 
            "models",
            "logs"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {directory}")
    
    def get_database_url(self) -> str:
        """Get database connection URL."""
        return f"sqlite:///{self.database.sqlite_path}"


# Global configuration instance
config = Config()