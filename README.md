# AI Knowledge Assistant

A production-ready AI-powered document search and chat assistant, built with FastAPI, Gemini AI, and ChromaDB.

## Features

- **Document Search**: Query internal documents (PDFs, PPTs, training decks) with semantic search
- **Chat Interface**: Conversational AI assistant with message history
- **Context Management**: Smart context handling with metadata awareness
- **Follow-up Suggestions**: AI-generated relevant follow-up questions
- **Production Logging**: Comprehensive structured logging for all operations
- **Document Processing**: Supports PDF, PPTX, DOCX, and image files with OCR
- **Vector Search**: ChromaDB vector store with sentence transformers embeddings

## Tech Stack

### Backend
- **Python 3.10+**
- **FastAPI** - Web framework
- **Gemini AI** - LLM for response generation
- **ChromaDB** - Vector database
- **LlamaIndex** - Document indexing and retrieval
- **Sentence Transformers** - Embedding model
- **Pydantic** - Data validation
- **Uvicorn** - ASGI server

### Frontend
- **React** - UI framework
- **React Markdown** - Markdown rendering
- **CSS Modules** - Styling
