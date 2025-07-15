import time
from typing import Tuple, List
from fastapi import HTTPException
from llama_index.core import VectorStoreIndex, Settings, get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from sentence_transformers import CrossEncoder
from query_utils import normalize_text

from settings import Settings as AppSettings
from schemas import SourceNode, SourceMetadata
from logging_utils import StructuredLogger

Settings.llm = None
class VectorService:
    """Vector search and retrieval service"""
    
    def __init__(self, logger: StructuredLogger):
        self.logger = logger
        self.index = None
        self.embed_model = None
        self._initialize()
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    def _initialize(self):
        """Initialize ChromaDB index and embedding model"""
        try:
            self.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
            
            # Initialize ChromaDB client and collection
            chroma_client = chromadb.PersistentClient(path=AppSettings.PERSIST_DIR)
            chroma_collection = chroma_client.get_collection(AppSettings.COLLECTION_NAME)
            
            # Create vector store and load index
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                embed_model=self.embed_model
            )
            
            self.logger.info(f"ChromaDB index loaded successfully", {
                "collection_count": chroma_collection.count()
            })
            
        except Exception as e:
            self.logger.error("Failed to initialize ChromaDB index", e)
            raise RuntimeError("Could not initialize search system")
        
    def extract_context(self, query: str, top_k: int = 3, response_mode: str = "compact") -> Tuple[str, List[SourceNode], str, List[dict]]:
        """Extract relevant context chunks using synthesizer"""
        if not self.index:
            raise HTTPException(status_code=500, detail="Search index not available")
        
        start_time = time.time()
        try:

            initial_top_k = top_k * AppSettings.INITIAL_RETRIEVAL_MULTIPLIER

            # Configure retriever
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=initial_top_k,
                verbose=True
            )
            
            # Configure synthesizer based on response mode
            response_mode_map = AppSettings.get_response_mode_map()
            current_synthesizer = get_response_synthesizer(
                response_mode=response_mode_map.get(response_mode, response_mode_map["compact"]),
                use_async=False,
                streaming=False
            )
            
            # Create query engine with custom synthesizer
            query_engine = RetrieverQueryEngine(
                retriever=retriever,
                response_synthesizer=current_synthesizer
            )
            
            # Execute query
            response = query_engine.query(query)
            duration = time.time() - start_time
            
            # Extract and enhance source nodes
            source_nodes = []
            chunks_for_logging = []
            
            if hasattr(response, 'source_nodes') and response.source_nodes:
                # Sort by score (descending) for better organization
                sorted_nodes = sorted(response.source_nodes, 
                                    key=lambda x: getattr(x, 'score', 0.0), 
                                    reverse=True)
                
                for i, node in enumerate(sorted_nodes):
                    # Extract metadata with error handling
                    node_metadata = {}
                    if hasattr(node, 'metadata') and node.metadata:
                        node_metadata = node.metadata
                    elif hasattr(node, 'node') and hasattr(node.node, 'metadata'):
                        node_metadata = node.node.metadata
                    
                    # Create simplified metadata object
                    simplified_metadata = SourceMetadata(
                        #source_type=node_metadata.get('source_type', 'document'),
                        doc_ref_id=node_metadata.get('DOC_REF_ID', ""),
                        score=float(getattr(node, 'score', 0.0)),

                        DOC_DESCRIPTION=node_metadata.get("DOC_DESCRIPTION", ""),
                        DOC_TITLE=node_metadata.get("DOC_TITLE", ""),
                        DOC_DESCRIPTION_FORMATTED=node_metadata.get("DOC_DESCRIPTION_FORMATTED", ""),
                        TAGS=node_metadata.get("TAGS", ""),
                        PRESENTATION_DATE=node_metadata.get("PRESENTATION_DATE", ""),
                        DOC_MODULE=node_metadata.get("DOC_MODULE", ""),
                        PRESENTATION_LINK=node_metadata.get("PRESENTATION_LINK", ""),
                        PRESENTER_1_NAME=node_metadata.get("PRESENTER_1_NAME", ""),
                    )


                    
                    # Get node text with error handling
                    node_text = ""
                    if hasattr(node, 'text'):
                        node_text = node.text
                    elif hasattr(node, 'node') and hasattr(node.node, 'text'):
                        node_text = node.node.text
                    elif hasattr(node, 'get_content'):
                        node_text = node.get_content()
                    
                    # Create source node object
                    source_node = SourceNode(
                        text=node_text,
                        score=float(getattr(node, 'score', 0.0)),
                        metadata=simplified_metadata,
                        node_id=getattr(node, 'node_id', f"node_{i+1}")
                    )
                    
                    source_nodes.append(source_node)
                    
                    # Add chunk for logging
                    chunks_for_logging.append({
                        "chunk_id": i + 1,
                        "presention_link": simplified_metadata.PRESENTATION_LINK,
                        # "doc_ref_id": simplified_metadata.doc_ref_id,
                        "score": simplified_metadata.score,
                        "text_length": len(node_text),
                        "text_preview": node_text[:200] + "..." if len(node_text) > 200 else node_text
                    })

            if source_nodes:
                # Rerank the nodes
                source_nodes = self._rerank_nodes(query, source_nodes, top_k)
            
            # Prepare context with inline metadata
            context_with_metadata = self._prepare_context_with_metadata(source_nodes)
            
            self.logger.info("Context extraction completed", {
                "query": query,
                "duration": duration,
                "context_length": len(context_with_metadata),
                "source_count": len(source_nodes),
                "synthesis_method": response_mode,
            })
            
            return context_with_metadata, source_nodes, response_mode, chunks_for_logging
            
        except Exception as e:
            self.logger.error("Context extraction failed", e, {"query": query})
            raise HTTPException(status_code=500, detail="Failed to extract context with metadata")
        
    def _rerank_nodes(self, query: str, nodes: List[SourceNode], top_k: int) -> List[SourceNode]:
        """Rerank nodes using cross-encoder"""
        try:
            # Create pairs for reranking
            pairs = [(query, node.text) for node in nodes]
            
            # Get scores from reranker
            scores = self.reranker.predict(pairs)
            
            # Combine nodes with scores
            scored_nodes = list(zip(nodes, scores))
            print("Reranking pairs:", scored_nodes)
            
            # Sort by score descending
            scored_nodes.sort(key=lambda x: x[1], reverse=True)
            
            # Take top_k and update their scores
            result = []
            for node, new_score in scored_nodes[:top_k]:
                # Update the score in metadata
                node.metadata.score = float(new_score)
                result.append(node)

            print("Reranked nodes:", result)    
            
            return result
        except Exception as e:
            self.logger.error("Reranking failed", e)
            return nodes[:top_k]  # Fallback to original top_k
    
    def _prepare_context_with_metadata(self, source_nodes: List[SourceNode]) -> str:
        """Build context string with inline metadata blocks before each chunk"""
        if not source_nodes:
            return "No relevant context found."
        
        context_parts = []
        
        for i, node in enumerate(source_nodes, 1):
            metadata = node.metadata
            
            chunk_text = normalize_text(node.text)

            # full_metadata = {}
            # if hasattr(node, 'node') and hasattr(node.node, 'metadata'):
            #     full_metadata = node.node.metadata
            # elif hasattr(node, 'metadata'):
            #     full_metadata = node.metadata
            
            # Build simplified inline metadata block
            metadata_block = f"""[METADATA]
                PRESENTATION_LINK: {getattr(metadata, 'PRESENTATION_LINK', 'N/A')}
                SCORE: {metadata.score:.3f}
                TITLE: {getattr(metadata, 'DOC_TITLE', 'N/A')}
                DESCRIPTION: {getattr(metadata, 'DOC_DESCRIPTION', 'N/A')}
                DESCRIPTION_FORMATTED: {getattr(metadata, 'DOC_DESCRIPTION_FORMATTED', 'N/A')}
                MODULE: {getattr(metadata, 'DOC_MODULE', 'N/A')}
                PRESENTATION_DATE: {getattr(metadata, 'PRESENTATION_DATE', 'N/A')}
                TAGS: {getattr(metadata, 'TAGS', 'N/A')}
                [/METADATA]
                """
            
            # Clean and format the content
            chunk_text = node.text.strip()
            if len(chunk_text) > AppSettings.MAX_CONTEXT_LENGTH:
                chunk_text = chunk_text[:AppSettings.MAX_CONTEXT_LENGTH] + "\n... [Content truncated for length] ..."
            
            # Combine metadata block with content
            complete_chunk = f"{metadata_block}{chunk_text}"
            context_parts.append(complete_chunk)
        
        # Join all chunks with separators
        return "\n\n---\n\n".join(context_parts)