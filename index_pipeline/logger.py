# logger.py

import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

class JSONLogger:
    """Custom JSON logger for document processing"""
    
    def __init__(self, log_file_path: str, logger_name: str = "document_processor"):
        self.log_file_path = log_file_path
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        
        # Ensure log directory exists
        Path(log_file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Create file handler
        handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
        handler.setLevel(logging.INFO)
        
        # Remove default formatting - we'll handle JSON formatting manually
        handler.setFormatter(logging.Formatter('%(message)s'))
        
        # Clear existing handlers and add our handler
        self.logger.handlers.clear()
        self.logger.addHandler(handler)
        
        # Also add console handler for real-time feedback
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(console_handler)
    
    def _create_base_log_entry(self, event_type: str, **kwargs) -> Dict[str, Any]:
        """Create base log entry with common fields"""
        return {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            **kwargs
        }
    
    def log_file_processing_start(self, filename: str, file_path: str, file_type: str):
        """Log the start of file processing"""
        log_entry = self._create_base_log_entry(
            "file_processing_start",
            filename=filename,
            file_path=file_path,
            file_type=file_type
        )
        self._write_json_log(log_entry)
    
    def log_file_processing_success(self, filename: str, file_path: str, file_type: str, 
                                  text_length: int, word_count: int, output_txt_path: str):
        """Log successful file processing"""
        log_entry = self._create_base_log_entry(
            "file_processing_success",
            filename=filename,
            file_path=file_path,
            file_type=file_type,
            text_length=text_length,
            word_count=word_count,
            output_txt_path=output_txt_path,
            status="success"
        )
        self._write_json_log(log_entry)
    
    def log_file_processing_error(self, filename: str, file_path: str, file_type: str, error: str):
        """Log file processing error"""
        log_entry = self._create_base_log_entry(
            "file_processing_error",
            filename=filename,
            file_path=file_path,
            file_type=file_type,
            error=str(error),
            status="failed"
        )
        self._write_json_log(log_entry)
    
    def log_chunking_start(self, txt_filename: str, txt_path: str):
        """Log the start of chunking process"""
        log_entry = self._create_base_log_entry(
            "chunking_start",
            txt_filename=txt_filename,
            txt_path=txt_path
        )
        self._write_json_log(log_entry)
    
    def log_chunking_success(self, txt_filename: str, txt_path: str, chunks_created: int, doc_ref_id: str):
        """Log successful chunking"""
        log_entry = self._create_base_log_entry(
            "chunking_success",
            txt_filename=txt_filename,
            txt_path=txt_path,
            chunks_created=chunks_created,
            doc_ref_id=doc_ref_id,
            status="success"
        )
        self._write_json_log(log_entry)
    
    def log_chunking_error(self, txt_filename: str, txt_path: str, error: str):
        """Log chunking error"""
        log_entry = self._create_base_log_entry(
            "chunking_error",
            txt_filename=txt_filename,
            txt_path=txt_path,
            error=str(error),
            status="failed"
        )
        self._write_json_log(log_entry)
    
    def log_indexing_start(self, total_nodes: int, total_files: int):
        """Log the start of vector indexing"""
        log_entry = self._create_base_log_entry(
            "indexing_start",
            total_nodes=total_nodes,
            total_files_processed=total_files
        )
        self._write_json_log(log_entry)
    
    def log_indexing_success(self, total_nodes: int, total_files: int, elapsed_time: float, 
                           embed_model: str, persist_dir: str):
        """Log successful indexing completion"""
        log_entry = self._create_base_log_entry(
            "indexing_success",
            total_nodes=total_nodes,
            total_files_processed=total_files,
            elapsed_time_seconds=elapsed_time,
            embed_model=embed_model,
            persist_dir=persist_dir,
            status="success"
        )
        self._write_json_log(log_entry)
    
    def log_indexing_error(self, error: str):
        """Log indexing error"""
        log_entry = self._create_base_log_entry(
            "indexing_error",
            error=str(error),
            status="failed"
        )
        self._write_json_log(log_entry)
    
    def log_summary(self, total_files: int, successful_files: int, failed_files: int, 
                   total_chunks: int, total_processing_time: float):
        """Log processing summary"""
        log_entry = self._create_base_log_entry(
            "processing_summary",
            total_files=total_files,
            successful_files=successful_files,
            failed_files=failed_files,
            total_chunks=total_chunks,
            total_processing_time_seconds=total_processing_time,
            success_rate=f"{(successful_files/total_files)*100:.1f}%" if total_files > 0 else "0%"
        )
        self._write_json_log(log_entry)
    
    def _write_json_log(self, log_entry: Dict[str, Any]):
        """Write JSON log entry to file"""
        try:
            json_line = json.dumps(log_entry, ensure_ascii=False, default=str)
            # Get the file handler (first handler is file, second is console)
            file_handler = self.logger.handlers[0]
            file_handler.stream.write(json_line + '\n')
            file_handler.stream.flush()
        except Exception as e:
            self.logger.error(f"Failed to write log entry: {e}")

    def log_file_complete(self, filename: str, file_path: str, file_type: str,
                        text_length: int, word_count: int, output_txt_path: str,
                        chunks_created: int, doc_ref_id: str):
        """Consolidated log for successful file processing with chunking info"""
        log_entry = self._create_base_log_entry(
            "file_processing_complete",
            filename=filename,
            file_path=file_path,
            file_type=file_type,
            text_length=text_length,
            word_count=word_count,
            output_txt_path=output_txt_path,
            chunks_created=chunks_created,
            doc_ref_id=doc_ref_id,
            status="success"
        )
        self._write_json_log(log_entry)
