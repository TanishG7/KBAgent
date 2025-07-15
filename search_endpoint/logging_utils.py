import logging
import json
import sys
from typing import Dict, Any, Optional
from schemas import RequestLog

class StructuredLogger:
    """Structured logger for production use"""
    
    def __init__(self, log_file: str = "document_search.json", log_level: str = "INFO"):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # File handler for JSON logs
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(file_formatter)
        
        # Console handler for structured logs
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def log_request(self, request_log: RequestLog):
        """Log complete request with structured JSON"""
        log_entry = {
            "log_type": "request_log",
            "request_id": request_log.request_id,
            "timestamp": request_log.timestamp.isoformat(),
            "query": request_log.query,
            "cleaned_query": request_log.cleaned_query,
            "top_k": request_log.top_k,
            "response_mode": request_log.response_mode,
            # "context": request_log.context,
            "chunks": request_log.chunks,
            "gemini_input": request_log.gemini_input,
            "gemini_output": request_log.gemini_output,
            "final_response": request_log.final_response,
            "processing_time": request_log.processing_time,
            "success": request_log.success,
            "error": request_log.error
        }
        
        self.logger.info(json.dumps(log_entry, ensure_ascii=False))
    
    def info(self, message: str, extra_data: Optional[Dict[str, Any]] = None):
        """Log info message with optional structured data"""
        if extra_data:
            log_entry = {"message": message, "data": extra_data}
            self.logger.info(json.dumps(log_entry, ensure_ascii=False))
        else:
            self.logger.info(message)
    
    def error(self, message: str, error: Exception = None, extra_data: Optional[Dict[str, Any]] = None):
        """Log error message with optional exception and structured data"""
        log_entry = {
            "message": message,
            "error": str(error) if error else None,
            "error_type": type(error).__name__ if error else None,
            "data": extra_data
        }
        self.logger.error(json.dumps(log_entry, ensure_ascii=False))