#!/bin/bash

# RAG System Demo Script
# This script demonstrates the RAG system functionality

set -e

echo "🚀 RAG System Demo Script"
echo "=========================="

# Configuration
EMBEDDING_SERVICE_URL="http://localhost:8000"
ORCHESTRATOR_URL="http://localhost:5000"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to make HTTP requests with error handling
make_request() {
    local method=$1
    local url=$2
    local data=$3
    local description=$4
    
    echo -e "${BLUE}📡 ${description}${NC}"
    echo "   ${method} ${url}"
    
    if [[ -z "$data" ]]; then
        response=$(curl -s -w "\nHTTP_CODE:%{http_code}" -X ${method} "${url}")
    else
        response=$(curl -s -w "\nHTTP_CODE:%{http_code}" -X ${method} "${url}" \
            -H "Content-Type: application/json" \
            -d "${data}")
    fi
    
    http_code=$(echo "$response" | tail -n1 | cut -d: -f2)
    response_body=$(echo "$response" | sed '$d')
    
    if [[ $http_code -ge 200 && $http_code -lt 300 ]]; then
        echo -e "${GREEN}✅ Success (HTTP $http_code)${NC}"
        echo "$response_body" | jq . 2>/dev/null || echo "$response_body"
    else
        echo -e "${RED}❌ Error (HTTP $http_code)${NC}"
        echo "$response_body"
        return 1
    fi
    
    echo ""
}

# Check if services are running
check_services() {
    echo -e "${YELLOW}🔍 Checking services...${NC}"
    
    # Check embedding service
    if curl -s "${EMBEDDING_SERVICE_URL}/health" > /dev/null; then
        echo -e "${GREEN}✅ Embedding service is running${NC}"
    else
        echo -e "${RED}❌ Embedding service is not responding${NC}"
        echo "   Make sure to run: docker-compose up -d"
        exit 1
    fi
    
    # Check orchestrator
    if curl -s "${ORCHESTRATOR_URL}/health" > /dev/null; then
        echo -e "${GREEN}✅ Orchestrator is running${NC}"
    else
        echo -e "${RED}❌ Orchestrator is not responding${NC}"
        echo "   Make sure to run: docker-compose up -d"
        exit 1
    fi
    
    echo ""
}

# Step 1: Health checks
health_checks() {
    echo -e "${YELLOW}🏥 Step 1: Health Checks${NC}"
    
    make_request "GET" "${EMBEDDING_SERVICE_URL}/health" "" "Checking embedding service health"
    make_request "GET" "${ORCHESTRATOR_URL}/health" "" "Checking orchestrator health"
}

# Step 2: Ingest sample documents
ingest_documents() {
    echo -e "${YELLOW}📚 Step 2: Ingesting Sample Documents${NC}"
    
    # Single document ingestion
    single_doc='{
        "id": "demo_doc_1",
        "text": "Artificial Intelligence is transforming how we work and live. Machine learning algorithms can analyze vast amounts of data to find patterns and make predictions. Deep learning, a subset of ML, uses neural networks with multiple layers to solve complex problems."
    }'
    
    make_request "POST" "${EMBEDDING_SERVICE_URL}/embed" "$single_doc" "Ingesting single document"
    
    # Bulk document ingestion
    bulk_docs='{
        "documents": [
            {
                "id": "nlp_doc",
                "text": "Natural Language Processing enables computers to understand and generate human language. It involves techniques like tokenization, part-of-speech tagging, named entity recognition, and sentiment analysis."
            },
            {
                "id": "computer_vision_doc", 
                "text": "Computer Vision allows machines to interpret visual information. It includes image classification, object detection, facial recognition, and image segmentation using convolutional neural networks."
            },
            {
                "id": "robotics_doc",
                "text": "Robotics combines AI with mechanical engineering to create autonomous systems. Modern robots use sensors, actuators, and AI algorithms to navigate environments and perform tasks."
            }
        ]
    }'
    
    make_request "POST" "${EMBEDDING_SERVICE_URL}/bulk_embed" "$bulk_docs" "Bulk ingesting documents"
}

# Step 3: Test similarity search
test_search() {
    echo -e "${YELLOW}🔍 Step 3: Testing Similarity Search${NC}"
    
    search_queries=(
        "machine learning algorithms"
        "natural language processing"
        "computer vision applications"
        "AI and robotics"
    )
    
    for query in "${search_queries[@]}"; do
        encoded_query=$(echo "$query" | sed 's/ /%20/g')
        make_request "GET" "${EMBEDDING_SERVICE_URL}/search?query=${encoded_query}&k=3" "" "Searching for: $query"
    done
}

# Step 4: Test RAG chat functionality
test_chat() {
    echo -e "${YELLOW}💬 Step 4: Testing RAG Chat${NC}"
    
    chat_queries=(
        '{
            "user_id": "demo_user_1",
            "query": "What is machine learning and how does it work?",
            "k": 3
        }'
        '{
            "user_id": "demo_user_1", 
            "query": "Explain the difference between NLP and computer vision",
            "k": 2
        }'
        '{
            "user_id": "demo_user_2",
            "query": "How are AI and robotics related?",
            "k": 3
        }'
    )
    
    for i in "${!chat_queries[@]}"; do
        make_request "POST" "${ORCHESTRATOR_URL}/chat" "${chat_queries[$i]}" "Chat query $((i+1))"
    done
}

# Step 5: Test batch chat
test_batch_chat() {
    echo -e "${YELLOW}📦 Step 5: Testing Batch Chat${NC}"
    
    batch_request='{
        "user_id": "demo_user_batch",
        "queries": [
            "What are neural networks?",
            "How does deep learning work?",
            "What is the future of AI?"
        ],
        "k": 2
    }'
    
    make_request "POST" "${ORCHESTRATOR_URL}/chat/batch" "$batch_request" "Batch chat request"
}

# Step 6: Check statistics
check_stats() {
    echo -e "${YELLOW}📊 Step 6: Checking System Statistics${NC}"
    
    make_request "GET" "${EMBEDDING_SERVICE_URL}/stats" "" "Getting embedding service statistics"
}

# Main execution
main() {
    echo -e "${GREEN}Starting RAG System Demo...${NC}"
    echo ""
    
    check_services
    health_checks
    ingest_documents
    test_search
    test_chat
    test_batch_chat
    check_stats
    
    echo -e "${GREEN}🎉 Demo completed successfully!${NC}"
    echo ""
    echo -e "${BLUE}📝 Summary:${NC}"
    echo "   • Ingested documents into vector store"
    echo "   • Tested similarity search functionality"
    echo "   • Demonstrated RAG chat capabilities"
    echo "   • Verified system health and statistics"
    echo ""
    echo -e "${YELLOW}💡 Next steps:${NC}"
    echo "   • Try the LoRA fine-tuning script: python scripts/train_adapter.py"
    echo "   • Ingest your own documents: python scripts/ingest_data.py data/"
    echo "   • Explore the API endpoints with Postman or curl"
}

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo -e "${YELLOW}⚠️  jq is not installed. JSON responses will not be formatted.${NC}"
    echo "   Install jq for better output formatting: brew install jq (macOS) or apt-get install jq (Ubuntu)"
    echo ""
fi

# Run the demo
main
