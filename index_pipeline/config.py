# config.py

import os
from pathlib import Path

class Config:
    """Configuration settings for the document indexing system"""
    
    # Directory paths
    RAW_DOCS_FOLDER = "../kbrawdocs"
    DOCS_FOLDER = "../docs"
    PERSIST_DIR = "../storage_chroma"
    LOGS_DIR = "../logs"

    BASE_DIR = Path(__file__).parent.resolve()

    
    PROCESSED_FILES_LOG = os.path.join(BASE_DIR, "processed_files.log")
    
    # ChromaDB settings
    COLLECTION_NAME = "document_collection"
    CHROMA_METADATA = {"hnsw:space": "cosine"}
    
    # Embedding model
    EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Chunking settings
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50
    
    # Logging settings
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "json"
    
    # Processing settings
    MAX_SAMPLE_PRINT = 3
    OCR_DPI = 300
    
    # Tesseract path (adjust for your system)
    TESSERACT_CMD = r'/usr/bin/tesseract'
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        directories = [
            cls.DOCS_FOLDER,
            cls.PERSIST_DIR,
            cls.LOGS_DIR
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_log_file_path(cls, log_type="main"):
        """Get the path for log files"""
        return os.path.join(cls.LOGS_DIR, f"{log_type}_{Config.get_timestamp()}.json")
    
    @staticmethod
    def get_timestamp():
        """Get current timestamp for file naming"""
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")