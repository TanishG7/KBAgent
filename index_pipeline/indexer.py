# indexer.py

import os
import time
import shutil
from typing import List, Optional
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import BaseNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

from config import Config
from logger import JSONLogger

class VectorIndexer:
    """Handles vector indexing with ChromaDB"""
    
    def __init__(self, logger: JSONLogger):
        self.logger = logger
        self.embed_model = HuggingFaceEmbedding(model_name=Config.EMBED_MODEL_NAME)
        self.vector_store = None
        self.index = None
    
    def setup_chroma_store(self, clear_existing: bool = False) -> bool:
        """
        Set up ChromaDB vector store
        
        Args:
            clear_existing: If True, will delete existing store. If False, will reuse existing store.
        
        Returns:
            True if setup successful, False otherwise
        """
        try:
            # Only clear if requested
            if clear_existing and os.path.exists(Config.PERSIST_DIR):
                print(f"Removing existing Chroma store at {Config.PERSIST_DIR}")
                try:
                    shutil.rmtree(Config.PERSIST_DIR)
                except Exception as e:
                    print(f"Error removing existing store: {e}")
                    return False
            
            # Create directory if it doesn't exist
            os.makedirs(Config.PERSIST_DIR, exist_ok=True)
            
            # Initialize Chroma client
            try:
                chroma_client = chromadb.PersistentClient(
                    path=Config.PERSIST_DIR,
                    settings=chromadb.Settings(anonymized_telemetry=False)
                )
            except Exception as e:
                print(f"Error creating Chroma client: {e}")
                return False
            
            # Add embedding model to metadata
            metadata = Config.CHROMA_METADATA.copy()
            metadata["embedding_model"] = Config.EMBED_MODEL_NAME
            
            # Get or create collection with more robust error handling
            try:
                # First try to get the collection
                chroma_collection = chroma_client.get_collection(Config.COLLECTION_NAME)
                print(f"Using existing Chroma collection '{Config.COLLECTION_NAME}'")
            except Exception as e:
                # If collection doesn't exist, create it
                print(f"Collection not found, creating new one: {Config.COLLECTION_NAME}")
                try:
                    chroma_collection = chroma_client.create_collection(
                        name=Config.COLLECTION_NAME,
                        metadata=metadata
                    )
                    print(f"Created new Chroma collection '{Config.COLLECTION_NAME}'")
                except Exception as create_error:
                    print(f"Error creating collection: {create_error}")
                    return False
            
            # Create vector store
            self.vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            return True
            
        except Exception as e:
            self.logger.log_indexing_error(f"Failed to setup ChromaDB: {str(e)}")
            print(f"ChromaDB setup error details: {e}")
            return False
    
    def build_index(self, nodes: List[BaseNode], total_files: int) -> Optional[VectorStoreIndex]:
        """
        Build vector index from document nodes
        
        Args:
            nodes: List of document nodes to index
            total_files: Total number of files processed
            
        Returns:
            VectorStoreIndex if successful, None otherwise
        """
        if not nodes:
            self.logger.log_indexing_error("No nodes provided for indexing")
            return None
        
        
        try:
            # Setup storage context
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            
            # Build index
            start_time = time.time()
            
            self.index = VectorStoreIndex(
                nodes=nodes,
                embed_model=self.embed_model,
                storage_context=storage_context,
                show_progress=True
            )
            
            elapsed_time = time.time() - start_time
            
            # Log successful indexing
            self.logger.log_indexing_success(
                total_nodes=len(nodes),
                total_files=total_files,
                elapsed_time=elapsed_time,
                embed_model=Config.EMBED_MODEL_NAME,
                persist_dir=os.path.abspath(Config.PERSIST_DIR)
            )
            
            return self.index
            
        except Exception as e:
            self.logger.log_indexing_error(str(e))
            return None
    
    def print_vector_store_samples(self, sample_size: int = None):
        """Print samples from the vector store for debugging"""
        if sample_size is None:
            sample_size = Config.MAX_SAMPLE_PRINT
            
        try:
            if not self.vector_store:
                print("Vector store not initialized")
                return
                
            collection = self.vector_store._collection
            count = collection.count()
            print(f"\nTotal vectors stored in Chroma: {count}")
            
            if count == 0:
                print("No vectors found in the store")
                return
            
            results = collection.peek(limit=sample_size)
            
            for i, (doc_id, doc_text, metadata) in enumerate(zip(
                results['ids'], results['documents'], results['metadatas']
            )):
                print(f"\nVector {i+1}")
                print(f"ID: {doc_id}")
                print(f"Text preview: {doc_text[:150]}...")
                print(f"Metadata: {metadata}")
                
        except Exception as e:
            print(f"Error printing Chroma store samples: {e}")
    
    def get_index_stats(self) -> dict:
        """Get statistics about the created index"""
        if not self.vector_store:
            return {"error": "Vector store not initialized"}
        
        try:
            collection = self.vector_store._collection
            count = collection.count()
            
            return {
                "total_vectors": count,
                "embedding_model": Config.EMBED_MODEL_NAME,
                "persist_directory": Config.PERSIST_DIR,
                "collection_name": Config.COLLECTION_NAME
            }
        except Exception as e:
            return {"error": str(e)}