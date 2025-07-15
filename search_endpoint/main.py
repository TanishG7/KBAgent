import time
import uuid
from datetime import datetime
import socket
from contextlib import closing
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from settings import Settings as AppSettings
from schemas import SearchRequest, SearchResponse, RequestLog , ChatSearchRequest, ChatSearchResponse, ChatMessage
from vector_service import VectorService
from ai_service import AIService
from logging_utils import StructuredLogger
from query_utils import QueryProcessor

# previous_context = ""
# previous_suggestions = []
# Initialize services
logger = StructuredLogger(AppSettings.LOG_FILE, AppSettings.LOG_LEVEL)
vector_service = VectorService(logger)
ai_service = AIService(logger)

app = FastAPI(title="Production Document Search API with ChromaDB")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def check_port(port: int) -> bool:
    """Check if a port is available"""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        return sock.connect_ex(('localhost', port)) != 0

@app.post("/ai/api/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    
    """Enhanced document search with structured logging"""
    request_id = str(uuid.uuid4())
    start_time = time.time()
    timestamp = datetime.now()
    
    try:
        # Clean query
        cleaned_query = QueryProcessor.clean_query(request.question)
        
        if request.is_follow_up and request.previous_context:
            context = request.previous_context
            source_nodes = []
            synthesis_method = "follow_up"
            chunks = []
        else:
            # Normal retrieval
            context, source_nodes, synthesis_method, chunks = vector_service.extract_context(
                cleaned_query, request.top_k, request.response_mode
            )
        
        # Generate AI response
        ai_response, gemini_input, gemini_output = ai_service.generate_response(
            request.question, context
        )
        
        
        processing_time = time.time() - start_time
        
        # Prepare final response
        final_response = SearchResponse(
            answer=ai_response.answer,
            context=context,
            suggestions=ai_response.suggestions,
            was_context_valid=ai_response.was_context_valid,
            confidence_score=ai_response.confidence_score,
            success=True,
            ai_used="gemini",
            processing_time=processing_time,
            source_nodes=source_nodes,
            synthesis_method=synthesis_method,
            total_sources=len(source_nodes)
        )
        
        # Log complete request
        request_log = RequestLog(
            request_id=request_id,
            timestamp=timestamp,
            query=request.question,
            cleaned_query=cleaned_query,
            top_k=request.top_k,
            response_mode=request.response_mode,
            context=context,
            chunks=chunks,
            gemini_input=gemini_input,
            gemini_output=gemini_output,
            final_response=final_response.dict(),
            processing_time=processing_time,
            success=True
        )
        
        logger.log_request(request_log)
        
        return final_response
        
    except Exception as e:
        processing_time = time.time() - start_time
        
        error_response = SearchResponse(
            answer=f"Search failed: {str(e)}",
            context="",
            suggestions=["Try rephrasing your question", "Check system status"],
            was_context_valid=False,
            confidence_score=0.0,
            success=False,
            ai_used="none",
            processing_time=processing_time,
            source_nodes=[],
            synthesis_method="none",
            total_sources=0
        )
        
        # Log failed request
        request_log = RequestLog(
            request_id=request_id,
            timestamp=timestamp,
            query=request.question,
            cleaned_query=QueryProcessor.clean_query(request.question),
            top_k=request.top_k,
            response_mode=request.response_mode,
            context="",
            chunks=[],
            gemini_input={},
            gemini_output={},
            final_response=error_response.dict(),
            processing_time=processing_time,
            success=False,
            error=str(e)
        )
        
        logger.log_request(request_log)
        
        return error_response


@app.post("/ai/api/search-chat", response_model=ChatSearchResponse)
async def search_chat(request: ChatSearchRequest):
    """Enhanced stateless chat search with context management"""
    request_id = str(uuid.uuid4())
    start_time = time.time()
    timestamp = datetime.now()
    #context_refreshed = False
    
    try:
        # Clean query
        cleaned_query = QueryProcessor.clean_query(request.question)
        
        # Determine if we need new context
        needs_new_context = _should_fetch_new_context(request)
        
        if needs_new_context:
            # Fetch new context
            context, source_nodes, synthesis_method, chunks = vector_service.extract_context(
                cleaned_query, request.top_k, request.response_mode
            )
            #context_refreshed = True
        else:
            # Use existing context
            context = request.previous_context or ""
            source_nodes = []
            synthesis_method = "reuse_context"
            chunks = []
        
        # Generate AI response with full message history
        ai_response, gemini_input, gemini_output = ai_service.generate_chat_response(
            message_history=request.message_history,
            current_question=request.question,
            context=context
        )

        was_context_valid_old_key = ai_response.was_context_valid

        was_context_valid_new_key = False
        
        # If context was invalid and we haven't refreshed yet, fetch new context
        if not ai_response.was_context_valid:
            logger.info("Context invalid, fetching new context", {"query": cleaned_query})
            context, source_nodes, synthesis_method, chunks = vector_service.extract_context(
                cleaned_query, request.top_k, request.response_mode
            )
            #context_refreshed = True

            # Retry with new context
            ai_response, gemini_input, gemini_output = ai_service.generate_chat_response(
                message_history=request.message_history,
                current_question=request.question,
                context=context
            )
            was_context_valid_new_key = ai_response.was_context_valid

        processing_time = time.time() - start_time
        
        # Prepare final response
        final_response = ChatSearchResponse(
            answer=ai_response.answer,
            context=context,
            suggestions=ai_response.suggestions,
            was_context_valid=ai_response.was_context_valid,
            was_context_valid_new_key=was_context_valid_new_key,
            was_context_valid_old_key=was_context_valid_old_key,
            confidence_score=ai_response.confidence_score,
            success=True,
            ai_used="gemini",
            processing_time=processing_time,
            source_nodes=source_nodes,
            synthesis_method=synthesis_method,
            total_sources=len(source_nodes),
            #conversation_context=conversation_context,
            #context_refreshed=context_refreshed
        )
        
        # Log complete request
        request_log = RequestLog(
            request_id=request_id,
            timestamp=timestamp,
            query=request.question,
            cleaned_query=cleaned_query,
            top_k=request.top_k,
            response_mode=request.response_mode,
            chunks=chunks,
            gemini_input=gemini_input,
            gemini_output=gemini_output,
            final_response=final_response.dict(),
            processing_time=processing_time,
            success=True,
        )
        
        logger.log_request(request_log)
        return final_response
        
    except Exception as e:
        processing_time = time.time() - start_time
        
        error_response = ChatSearchResponse(
            answer=f"I apologize, but I encountered an error: {str(e)}",
            context="",
            suggestions=["Try rephrasing your question", "Check your connection", "Contact support if issue persists"],
            was_context_valid=False,
            was_context_valid_new_key=False,
            was_context_valid_old_key=False,
            confidence_score=0.0,
            success=False,
            ai_used="none",
            processing_time=processing_time,
            source_nodes=[],
            synthesis_method="none",
            total_sources=0,
            conversation_context="",
            context_refreshed=False
        )
        
        # Log failed request
        request_log = RequestLog(
            request_id=request_id,
            timestamp=timestamp,
            query=request.question,
            cleaned_query=QueryProcessor.clean_query(request.question),
            top_k=request.top_k,
            response_mode=request.response_mode,
            chunks=[],
            gemini_input={},
            gemini_output={},
            final_response=error_response.dict(),
            processing_time=processing_time,
            success=False,
            error=str(e)
        )
        
        logger.log_request(request_log)
        return error_response


def _should_fetch_new_context(request: ChatSearchRequest) -> bool:
    """Determine if we need to fetch new context based on conversation state"""
    
    # Always fetch for first interaction (no history)
    if not request.message_history:
        return True
    
    
    # Check if we have previous context
    if not request.previous_context:
        return True
        
    # Check if context is getting stale (more than 5 exchanges)
    # if len(request.message_history) > 10:  # 5 exchanges = 10 messages
    #     return True
    
    return False

@app.get("/ai/api/health")
async def health_check():
    """Health check endpoint with service status"""
    return {
        "status": "healthy",
        "services": {
            "chromadb": vector_service.index is not None,
            "gemini": ai_service.client is not None
        },
        "timestamp": time.time()
    }

if __name__ == "__main__":
    import uvicorn
    
    PORT = 8000
    if not check_port(PORT):
        logger.info(f"Port {PORT} in use, trying 8001")
        PORT = 8001
    
    logger.info(f"Starting Production API on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)