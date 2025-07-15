import os
from typing import Dict, Any

class Settings:
    """Application configuration settings"""
    
    # Storage configuration
    PERSIST_DIR = "../storage_chroma"
    COLLECTION_NAME = "document_collection"
    
    # API configuration
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyAYX1Ii7U8RR7ZIOj9zbxr7_iLfH_4jANQ")
    GEMINI_MODEL = "gemini-2.5-flash"
    
    INITIAL_RETRIEVAL_MULTIPLIER = 2
    
    # Search configuration
    DEFAULT_TOP_K = 3
    DEFAULT_RESPONSE_MODE = "compact"
    MAX_CONTEXT_LENGTH = 3000
    
    # Gemini configuration
    GEMINI_TEMPERATURE = 0.2
    GEMINI_MAX_OUTPUT_TOKENS = 1500
    GEMINI_TOP_P = 0.8
    GEMINI_TOP_K = 40
    
    # Logging configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = "../logs/document_search.log"
    
    @classmethod
    def get_response_mode_map(cls) -> Dict[str, Any]:
        from llama_index.core.response_synthesizers import ResponseMode
        return {
            "compact": ResponseMode.COMPACT,
            "tree_summarize": ResponseMode.TREE_SUMMARIZE,
            "accumulate": ResponseMode.ACCUMULATE,
            "simple_summarize": ResponseMode.SIMPLE_SUMMARIZE
        }