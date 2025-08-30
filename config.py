"""
Configuration file for Medical AI Chatbot

This file contains all configuration settings including API keys,
model parameters, and application settings.
"""

import os
from typing import Optional

# API Keys
PINECONE_API_KEY = ""
OPENAI_API_KEY = ""

# Pinecone Configuration
PINECONE_INDEX_NAME = "medicalbot"
PINECONE_DIMENSION = 384
PINECONE_METRIC = "cosine"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"

# OpenAI Configuration
OPENAI_MODEL = "gpt-3.5-turbo"
OPENAI_TEMPERATURE = 0.4
OPENAI_MAX_TOKENS = 500

# Document Processing Configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 20
MIN_CHUNK_SIZE = 50

# Embedding Model Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"  # Use "cuda" if GPU is available

# Flask Application Configuration
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 8080
FLASK_DEBUG = False
FLASK_THREADED = True

# Cache Configuration
CACHE_TYPE = "simple"
CACHE_DEFAULT_TIMEOUT = 300  # 5 minutes
CACHE_THRESHOLD = 1000

# Rate Limiting Configuration
RATE_LIMIT_DEFAULT = "200 per day"
RATE_LIMIT_CHAT = "10 per minute"

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = "medical_chatbot.log"

# Performance Configuration
MAX_INPUT_LENGTH = 1000
REQUEST_TIMEOUT = 30
INDEX_READY_TIMEOUT = 300

# Health Check Configuration
HEALTH_CHECK_INTERVAL = 60
HEALTH_CHECK_TIMEOUT = 10

# Error Handling Configuration
MAX_RETRIES = 3
RETRY_DELAY = 1

def get_config_value(key: str, default: Optional[str] = None) -> str:
    """
    Get configuration value with environment variable override support.
    
    Args:
        key (str): Configuration key
        default: Default value if not found
        
    Returns:
        str: Configuration value
    """
    # Check environment variable first
    env_key = f"MEDICAL_CHATBOT_{key}"
    if env_key in os.environ:
        return os.environ[env_key]
    
    # Return from config or default
    return getattr(config, key, default)

def validate_config() -> bool:
    """
    Validate that all required configuration values are set.
    
    Returns:
        bool: True if configuration is valid, False otherwise
    """
    required_keys = [
        'PINECONE_API_KEY',
        'OPENAI_API_KEY'
    ]
    
    for key in required_keys:
        if not get_config_value(key):
            print(f"Error: Required configuration '{key}' is not set")
            return False
    
    return True

# Environment-specific configurations
if os.getenv('FLASK_ENV') == 'production':
    FLASK_DEBUG = False
    LOG_LEVEL = "WARNING"
    CACHE_TYPE = "redis"  # Use Redis in production
elif os.getenv('FLASK_ENV') == 'development':
    FLASK_DEBUG = True
    LOG_LEVEL = "DEBUG"
    CACHE_TYPE = "simple"

# Note: In production, it's recommended to use environment variables
# You can also set these as environment variables:
# export MEDICAL_CHATBOT_PINECONE_API_KEY="your_actual_key"
# export MEDICAL_CHATBOT_OPENAI_API_KEY="your_actual_key"
