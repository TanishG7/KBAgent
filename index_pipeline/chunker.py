import uuid
from typing import List, Optional
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode

from config import Config
from logger import JSONLogger
import os

class DocumentChunker:
    """Handles document chunking with comprehensive logging"""
    
    def __init__(self, logger: JSONLogger):
        self.logger = logger
        self.chunker = SentenceSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            include_metadata=True,
            include_prev_next_rel=True
        )
    
    def chunk_document(self, txt_path: str, source_filename: str , doc_metadata: dict = None) -> Optional[List[BaseNode]]:
        """
        Chunk a text document into smaller pieces
        
        Args:
            txt_path: Path to the text file
            source_filename: Original source filename for metadata
            
        Returns:
            List of document nodes if successful, None otherwise
        """

        if not os.path.exists(txt_path):
            self.logger.log_chunking_error(source_filename, txt_path, "File not found")
            return None
    
        txt_filename = txt_path.split('/')[-1]  # Extract filename from path
        
        # Log chunking start
        # self.logger.log_chunking_start(txt_filename, txt_path)
        
        try:
            # Load the document
            reader = SimpleDirectoryReader(input_files=[txt_path])
            documents = reader.load_data()
            
            if not documents:
                error_msg = "No documents loaded from file"
                self.logger.log_chunking_error(txt_filename, txt_path, error_msg)
                return None
            
            # Generate unique document reference ID
            doc_ref_id = str(uuid.uuid4())
            
            # Process each document (usually just one for a single file)
            all_nodes = []
            for doc in documents:
                # Add metadata
                doc.metadata.update({
                    "DOC_DESCRIPTION": doc_metadata.get("DOC_DESCRIPTION", ""),
                    "DOC_TITLE": doc_metadata.get("DOC_TITLE", ""),
                    "DOC_DESCRIPTION_FORMATTED": doc_metadata.get("DOC_DESCRIPTION_FORMATTED", ""),
                    "TAGS": doc_metadata.get("TAGS", ""),
                    "PRESENTATION_DATE": doc_metadata.get("PRESENTATION_DATE", ""),
                    "DOC_MODULE": doc_metadata.get("DOC_MODULE", ""),
                    "PRESENTATION_LINK" : doc_metadata.get("PRESENTATION_LINK", ""),
                    "PRESENTER_1_NAME" : doc_metadata.get("PRESENTER_1_NAME", ""),
                    "DOC_REF_ID" : doc_ref_id,
                })
                
                # Create chunks
                nodes = self.chunker.get_nodes_from_documents([doc])
                all_nodes.extend(nodes)
            
            # Log successful chunking
            # chunks_created = len(all_nodes)
            # self.logger.log_chunking_success(txt_filename, txt_path, chunks_created, doc_ref_id)
            
            return all_nodes
            
        except Exception as e:
            self.logger.log_chunking_error(txt_filename, txt_path, str(e))
            return None
    
    def print_sample_nodes(self, nodes: List[BaseNode], sample_size: int = None):
        """Print sample of nodes for debugging"""
        if sample_size is None:
            sample_size = Config.MAX_SAMPLE_PRINT
            
        print(f"\nSample chunks (showing {min(len(nodes), sample_size)} of {len(nodes)}):")
        
        for i, node in enumerate(nodes[:sample_size]):
            print(f"\nChunk {i+1}")
            print(f"Text length: {len(node.text)}")
            print(f"Node ID: {node.node_id}")
            print(f"Metadata: {node.metadata}")
            print("Preview:")
            preview_text = node.text[:200] + "..." if len(node.text) > 200 else node.text
            print(preview_text)