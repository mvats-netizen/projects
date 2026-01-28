"""
Configuration Loader

Loads settings from config/secrets.env file.

Usage:
    >>> from src.config import config
    >>> 
    >>> # Access Databricks settings
    >>> print(config.DATABRICKS_HOST)
    >>> 
    >>> # Access embedding settings
    >>> print(config.EMBEDDING_PROVIDER)
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def find_project_root() -> Path:
    """Find the project root directory."""
    current = Path(__file__).parent
    for _ in range(5):
        if (current / "config").exists():
            return current
        current = current.parent
    return Path(__file__).parent.parent


def load_env_file(file_path: Path) -> dict:
    """Load environment variables from a file."""
    env_vars = {}
    
    if not file_path.exists():
        logger.warning(f"Config file not found: {file_path}")
        return env_vars
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Remove quotes
                if (value.startswith('"') and value.endswith('"')) or \
                   (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]
                
                env_vars[key] = value
    
    return env_vars


@dataclass
class Config:
    """Application configuration."""
    
    # Databricks
    DATABRICKS_HOST: Optional[str] = None
    DATABRICKS_TOKEN: Optional[str] = None
    DATABRICKS_HTTP_PATH: Optional[str] = None
    DATABRICKS_CATALOG: str = "main"
    DATABRICKS_SCHEMA: str = "default"
    
    # Embeddings
    EMBEDDING_PROVIDER: str = "local"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    # API Keys
    GOOGLE_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    
    @classmethod
    def load(cls, env_file: Optional[str] = None) -> "Config":
        """
        Load configuration from file and environment.
        
        Priority: Environment variables > .env files > defaults
        
        Searches for .env files in order:
        1. Explicit env_file path
        2. content-curations/.env
        3. content-curations/config/secrets.env
        4. projects/.env (parent directory)
        """
        file_vars = {}
        
        if env_file:
            file_vars = load_env_file(Path(env_file))
        else:
            # Search for .env files in order of priority
            # User's .env in parent directory takes priority
            project_root = find_project_root()
            search_paths = [
                project_root.parent / ".env",  # projects/.env - USER'S CREDENTIALS (highest priority)
                project_root / ".env",
                project_root / "config" / "secrets.env",
                project_root / "config" / "databricks.env",
            ]
            
            for path in search_paths:
                if path.exists():
                    loaded = load_env_file(path)
                    # Only load values that aren't placeholders
                    for key, value in loaded.items():
                        if key not in file_vars:
                            # Skip placeholder values
                            if "your_" not in value.lower() and "your-" not in value.lower():
                                file_vars[key] = value
                                logger.info(f"Loaded {key} from {path}")
        
        # Build config with priority: env > file > default
        def get_value(key: str, default: str = None):
            return os.getenv(key) or file_vars.get(key) or default
        
        return cls(
            DATABRICKS_HOST=get_value("DATABRICKS_HOST"),
            DATABRICKS_TOKEN=get_value("DATABRICKS_TOKEN"),
            DATABRICKS_HTTP_PATH=get_value("DATABRICKS_HTTP_PATH"),
            DATABRICKS_CATALOG=get_value("DATABRICKS_CATALOG", "main"),
            DATABRICKS_SCHEMA=get_value("DATABRICKS_SCHEMA", "default"),
            EMBEDDING_PROVIDER=get_value("EMBEDDING_PROVIDER", "local"),
            EMBEDDING_MODEL=get_value("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            GOOGLE_API_KEY=get_value("GOOGLE_API_KEY"),
            OPENAI_API_KEY=get_value("OPENAI_API_KEY"),
        )
    
    def is_databricks_configured(self) -> bool:
        """Check if Databricks credentials are set."""
        return bool(self.DATABRICKS_HOST and self.DATABRICKS_TOKEN)
    
    def is_gemini_configured(self) -> bool:
        """Check if Gemini API key is set."""
        return bool(self.GOOGLE_API_KEY)
    
    def is_openai_configured(self) -> bool:
        """Check if OpenAI API key is set."""
        return bool(self.OPENAI_API_KEY)
    
    def summary(self) -> str:
        """Return a summary of configuration status."""
        lines = [
            "Configuration Status:",
            f"  Databricks: {'✅ Configured' if self.is_databricks_configured() else '❌ Not configured'}",
            f"  Embeddings: {self.EMBEDDING_PROVIDER} ({self.EMBEDDING_MODEL})",
            f"  Gemini API: {'✅ Configured' if self.is_gemini_configured() else '⚪ Not set'}",
            f"  OpenAI API: {'✅ Configured' if self.is_openai_configured() else '⚪ Not set'}",
        ]
        return "\n".join(lines)


# Global config instance (lazy loaded)
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global config instance."""
    global _config
    if _config is None:
        _config = Config.load()
    return _config


# Convenience alias
config = property(lambda self: get_config())


# For direct access: from src.config import config
class ConfigProxy:
    """Proxy class to allow 'from src.config import config' syntax."""
    def __getattr__(self, name):
        return getattr(get_config(), name)


config = ConfigProxy()
