#!/bin/bash
# Unified setup and management script for all pipelines

set -euo pipefail

# Configuration
PYTHON_VERSION="3.10"
BASE_DIR=$(pwd)
VENV_DIR=".venv"
LOG_DIR="$BASE_DIR/logs"
STORAGE_DIR="$BASE_DIR/storage_chroma"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Common functions
create_dirs() {
    echo -e "${YELLOW}Creating directories...${NC}"
    mkdir -p "$LOG_DIR"
    mkdir -p "$STORAGE_DIR"
}

setup_python_venv() {
    local pipeline_dir=$1
    echo -e "${YELLOW}Setting up Python virtual environment in $pipeline_dir...${NC}"
    
    cd "$BASE_DIR/$pipeline_dir"
    python$PYTHON_VERSION -m venv $VENV_DIR
    source $VENV_DIR/bin/activate
    pip install --upgrade pip
}

install_python_deps() {
    local pipeline_dir=$1
    echo -e "${YELLOW}Installing Python dependencies for $pipeline_dir...${NC}"
    
    cd "$BASE_DIR/$pipeline_dir"
    source $VENV_DIR/bin/activate
    pip install -r requirements.txt
}

# Pipeline-specific functions
setup_search_endpoint() {
    echo -e "${GREEN}Setting up search endpoint...${NC}"
    
    setup_python_venv "search_endpoint"
    install_python_deps "search_endpoint"
    
    # Create search endpoint config
    cd "$BASE_DIR/search_endpoint"
    cat > .env <<EOL
GEMINI_API_KEY="your-api-key-here"
PERSIST_DIR="$STORAGE_DIR"
COLLECTION_NAME="document_collection"
LOG_FILE="$LOG_DIR/document_search.log"
EOL

    echo -e "${GREEN}Search endpoint setup complete${NC}"
}

setup_index_pipeline() {
    echo -e "${GREEN}Setting up index pipeline...${NC}"
    
    setup_python_venv "index_pipeline"
    install_python_deps "index_pipeline"
    
    # Create index pipeline config
    cd "$BASE_DIR/index_pipeline"
    cat > config.yaml <<EOL
storage:
  persist_dir: "$STORAGE_DIR"
  collection_name: "document_collection"
embedding:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
EOL

    echo -e "${GREEN}Index pipeline setup complete${NC}"
}

setup_frontend() {
    echo -e "${GREEN}Setting up frontend...${NC}"
    
    cd "$BASE_DIR/frontend"
    
    # Check if it's a Node.js or Python frontend
    if [ -f "package.json" ]; then
        echo "Detected Node.js frontend"
        npm install
    elif [ -f "requirements.txt" ]; then
        echo "Detected Python frontend"
        setup_python_venv "frontend"
        install_python_deps "frontend"
    else
        echo -e "${RED}No recognized frontend configuration found${NC}"
        exit 1
    fi

    echo -e "${GREEN}Frontend setup complete${NC}"
}

start_services() {
    echo -e "${GREEN}Starting all services...${NC}"
    
    # Start index pipeline in background
    cd "$BASE_DIR/index_pipeline"
    source $VENV_DIR/bin/activate
    nohup python main.py >> "$LOG_DIR/index_pipeline.log" 2>&1 &
    echo "Index pipeline started (PID: $!)"
    
    # Start search endpoint in background
    cd "$BASE_DIR/search_endpoint"
    source $VENV_DIR/bin/activate
    nohup uvicorn main:app --host 0.0.0.0 --port 8000 >> "$LOG_DIR/search_endpoint.log" 2>&1 &
    echo "Search endpoint started (PID: $!)"
    
    # Start frontend
    cd "$BASE_DIR/frontend"
    if [ -f "package.json" ]; then
        npm run dev
    elif [ -f "main.py" ]; then
        source $VENV_DIR/bin/activate
        python main.py
    fi
}

# Main menu
show_menu() {
    echo -e "\n${YELLOW}Document Search System Management${NC}"
    echo "1. Setup all pipelines"
    echo "2. Setup search endpoint only"
    echo "3. Setup index pipeline only"
    echo "4. Setup frontend only"
    echo "5. Start all services"
    echo "6. Exit"
    echo -n "Enter your choice [1-6]: "
}

# Handle user selection
while true; do
    show_menu
    read choice
    case $choice in
        1)
            create_dirs
            setup_search_endpoint
            setup_index_pipeline
            setup_frontend
            ;;
        2)
            create_dirs
            setup_search_endpoint
            ;;
        3)
            create_dirs
            setup_index_pipeline
            ;;
        4)
            setup_frontend
            ;;
        5)
            start_services
            ;;
        6)
            echo -e "${GREEN}Exiting...${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid option, please try again${NC}"
            ;;
    esac
done