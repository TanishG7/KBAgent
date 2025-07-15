import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import uuid

from config import Config
from logger import JSONLogger
from preprocess import DocumentProcessor
from chunker import DocumentChunker
from indexer import VectorIndexer

class LocalFileIndexingPipeline:
    def __init__(self, clear_existing: bool = False):
        """Initialize pipeline
        
        Args:
            clear_existing: If True, will clear existing index and process all files
        """
        # Create necessary directories
        Config.create_directories()

        if not os.path.exists(Config.RAW_DOCS_FOLDER):
            raise FileNotFoundError(f"Raw documents folder not found: {Config.RAW_DOCS_FOLDER}")
        
        # Initialize logger
        log_file_path = Config.get_log_file_path("indexing_pipeline")
        self.logger = JSONLogger(log_file_path, "local_file_indexing_pipeline")
        
        # Initialize components
        self.processor = DocumentProcessor(self.logger)
        self.chunker = DocumentChunker(self.logger)
        self.indexer = VectorIndexer(self.logger)

        self.stats = {
        "processing_start_time": None,
        "processing_end_time": None,
        "total_files": 0,
        "processed_files": 0,
        "skipped_files": 0,
        "successful_files": 0,
        "failed_files": 0,      # Ensure this exists
        "total_chunks": 0,
    }
        
        # File tracking
        self.processed_files = set()
        self.clear_existing = clear_existing
        if not clear_existing:
            self._load_processed_files()

    def _load_processed_files(self):
        """Load set of already processed files"""
        try:
            if os.path.exists(Config.PROCESSED_FILES_LOG):
                with open(Config.PROCESSED_FILES_LOG, 'r') as f:
                    self.processed_files = set(line.strip() for line in f if line.strip())
        except Exception as e:
            print(f"Warning: Could not load processed files log: {e}")

    def _save_processed_file(self, file_path: str):
        """Mark a file as processed"""
        self.processed_files.add(file_path)
        try:
            with open(Config.PROCESSED_FILES_LOG, 'a') as f:
                f.write(f"{file_path}\n")
        except Exception as e:
            print(f"Warning: Could not update processed files log: {e}")

    def run_pipeline(self) -> bool:
        print("\nStarting Local File Indexing Pipeline...\n")
        self.stats["processing_start_time"] = time.time()
        
        try:
            # Setup vector store (pass clear_existing flag)
            print(f"\nSetting up ChromaDB vector store...")
            if not self.indexer.setup_chroma_store(clear_existing=self.clear_existing):
                print("Failed to setup ChromaDB vector store")
                return False
            
            # Process files
            print("\nProcessing files from local folder...")
            raw_files = self._get_raw_files()
            
            for i, file_path in enumerate(raw_files):
                if not self.clear_existing and file_path in self.processed_files:
                    print(f"Skipping already processed file: {os.path.basename(file_path)}")
                    continue
                
                self.stats["total_files"] += 1
                filename = os.path.basename(file_path)
                print(f"\nProcessing ({i+1}/{len(raw_files)}): {filename}")
                
                if self._process_and_index_file(file_path):
                    self.stats["successful_files"] += 1
                    self._save_processed_file(file_path)
                else:
                    self.stats["failed_files"] += 1
                
                time.sleep(0.1)
            
            # Step 3: Print results and summary
            self._print_results()
            
            # Step 4: Log final summary
            self.stats["processing_end_time"] = time.time()
            total_time = self.stats["processing_end_time"] - self.stats["processing_start_time"]
            
            self.logger.log_summary(
                total_files=self.stats["total_files"],
                successful_files=self.stats["successful_files"],
                failed_files=self.stats["failed_files"],
                total_chunks=self.stats["total_chunks"],
                total_processing_time=total_time
            )
            
            print("\nDocument indexing pipeline completed successfully!")
            return True
            
        except Exception as e:
            self.logger.log_indexing_error(f"Pipeline failed: {str(e)}")
            print(f"Pipeline failed with error: {e}")
            return False
    
    def _get_raw_files(self) -> List[str]:
        """Get list of raw files to process"""
        try:
            files = []
            for filename in os.listdir(Config.RAW_DOCS_FOLDER):
                file_path = os.path.join(Config.RAW_DOCS_FOLDER, filename)
                if os.path.isfile(file_path):
                    files.append(file_path)
            return files
        except Exception as e:
            print(f"Error reading raw documents folder: {e}")
            return []
    
    def _process_and_index_file(self, file_path: str) -> bool:
        """
        Process a single file through the complete pipeline and index its chunks
        
        Args:
            file_path: Path to the raw file
            
        Returns:
            True if file was successfully processed and indexed, False otherwise
        """
        filename = os.path.basename(file_path)
        
        # Step 1: Convert file to text
        txt_path = self.processor.process_file(file_path, Config.DOCS_FOLDER)
        if not txt_path:
            print(f"Failed to process file: {filename}")
            return False
        
        # Step 2: Chunk the text document
        nodes = self.chunker.chunk_document(txt_path, filename, {})
        if not nodes:
            print(f"Failed to chunk file: {filename}")
            return False
        
        # Step 3: Index the chunks
        index = self.indexer.build_index(nodes, 1)  # We pass 1 for single file
        if not index:
            print(f"Failed to index file: {filename}")
            return False
        
        # Update statistics
        self.stats["total_chunks"] += len(nodes)
        
        # Log successful processing
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read()
            text_length = len(text)
            word_count = len(text.split())

        self.logger.log_file_complete(
            filename=filename,
            file_path=file_path,
            file_type=os.path.splitext(filename)[1],
            text_length=text_length,
            word_count=word_count,
            output_txt_path=txt_path,
            chunks_created=len(nodes),
            doc_ref_id=nodes[0].metadata["DOC_REF_ID"] if nodes else str(uuid.uuid4())
        )
        
        print(f"Successfully processed and indexed {filename} → {len(nodes)} chunks")
        return True
    
    def _print_results(self):
        """Print pipeline results and statistics"""
        print(f"\n" + "="*60)
        print("PROCESSING RESULTS")
        print("="*60)
        
        print(f"Total files found: {self.stats['total_files']}")
        print(f"Successfully processed: {self.stats['successful_files']}")
        print(f"Failed to process: {self.stats['failed_files']}")
        print(f"Total chunks created: {self.stats['total_chunks']}")
        
        if self.stats['total_files'] > 0:
            success_rate = (self.stats['successful_files'] / self.stats['total_files']) * 100
            print(f"Success rate: {success_rate:.1f}%")
        
        # Print vector store statistics
        print(f"\nVECTOR STORE INFORMATION")
        print("-" * 30)
        index_stats = self.indexer.get_index_stats()
        for key, value in index_stats.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        
        # Print sample vectors
        self.indexer.print_vector_store_samples()
        
        total_time = time.time() - self.stats["processing_start_time"]
        print(f"\nTotal processing time: {total_time:.2f} seconds")
        print(f"Logs saved to: {self.logger.log_file_path}")
        print("="*60)

def main():
    """Main entry point"""
    # To force reindexing of all files, pass clear_existing=True
    pipeline = LocalFileIndexingPipeline(clear_existing=False)  # Set to True to reindex everything
    success = pipeline.run_pipeline()
    
    if success:
        print("\nAll done! Your documents are now indexed and ready for search.")
    else:
        print("\nPipeline failed. Check the logs for detailed error information.")
    
    return success

if __name__ == "__main__":
    main()