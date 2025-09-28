# RAG System with LoRA Fine-tuning - AI Engineer Challenge

A Retrieval-Augmented Generation (RAG) system with MongoDB Atlas vector search, OpenAI integration, and LoRA adapter fine-tuning capabilities.

## 🏗️ Architecture

```
                    RAG PIPELINE
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│     Client      │    │   Orchestrator   │    │ Embedding API   │
│                 │◄──►│   (Node.js)      │◄──►│   (FastAPI)     │
│ • curl/API      │    │ • /chat endpoint │    │ • Jina AI       │
│ • Demo script   │    │ • OpenAI GPT     │    │ • Vector store  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   OpenAI API    │    │ MongoDB Atlas   │
                       │                 │    │                 │
                       │ • gpt-3.5-turbo │    │ • Vector search │
                       │ • Context-aware │    │ • Cosine sim    │
                       │ • Response gen  │    │ • 1024D vectors │
                       └─────────────────┘    └─────────────────┘

                    LORA FINE-TUNING (Independent)
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Training Data  │    │   LoRA Trainer   │    │  Trained Model  │
│                 │───►│                  │───►│                 │
│ • QA pairs      │    │ • PEFT library   │    │ • 1.6MB adapter │
│ • 8 examples    │    │ • DistilGPT-2    │    │ • Interactive   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📁 Project Structure

```
ai-engineer-challenge/
├── services/
│   ├── embedding_service/          # Python FastAPI service
│   │   ├── main.py                # FastAPI application
│   │   ├── models.py              # Pydantic models
│   │   ├── embedding_service.py   # Jina AI & local embeddings
│   │   ├── vector_store.py        # MongoDB vector operations
│   │   ├── requirements.txt       # Python dependencies
│   │   └── Dockerfile            # Container config
│   └── orchestrator/              # Node.js Express service
│       ├── src/
│       │   ├── app.js            # Main Express app
│       │   ├── embeddingClient.js # Embedding service client
│       │   ├── llmService.js     # OpenAI/LLM integration
│       │   └── logger.js         # Winston logging
│       ├── package.json          # Node.js dependencies
│       └── Dockerfile           # Container config
├── scripts/
│   ├── train_adapter.py         # Complete LoRA training & inference demo
│   ├── use_lora_model.py        # Interactive LoRA model usage script
│   ├── ingest_data.py           # Data ingestion pipeline
│   ├── quick_multi_vector_demo.py  # Simple multi-vector similarity demo
│   ├── test_multi_vector.py     # Comprehensive multi-vector testing
│   └── requirements.txt         # Python ML dependencies
├── data/
│   ├── sample_documents.json     # Demo dataset (20 docs)
│   ├── rag_explanation.txt      # RAG methodology
│   └── vector_databases.md      # Vector DB concepts
├── docs/                        # Additional documentation
├── docker-compose.yml           # Multi-service orchestration
├── demo.sh                      # Demo script with curl commands
├── .env.example                 # Environment template
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## 🚀 How to Run Locally

### Prerequisites
- Docker & Docker Compose
- Python 3.9+ (for LoRA training)

### 1. Environment Setup
```bash
# Clone and navigate to project
cd ai-engineer-challenge

# Set up environment variables (.env file)
OPENAI_API_KEY=your_openai_api_key_here      # Get from https://openai.com
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/  # MongoDB Atlas URI
MONGODB_DATABASE=rag_system
MONGODB_COLLECTION=embeddings
```

### 2. Start RAG System
```bash
# Start all services with Docker Compose
docker-compose -f docker-compose.atlas.yml up -d

# Check service health
curl http://localhost:8000/health  # Embedding service
curl http://localhost:5000/health  # Orchestrator

# View logs
docker-compose logs -f
```

### 3. Set Up LoRA Environment (Separate from Docker)
```bash
# Create Python virtual environment for LoRA
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install LoRA dependencies
pip install torch transformers peft datasets
```

### 4. Development with React Demo UI
```bash
# Start backend services
docker-compose up -d

# Install and run React demo (in a new terminal)
cd demo-ui
npm install
npm start

# Access at http://localhost:3000 (connects to backend on localhost:5000)
# Backend services run on: http://localhost:5000 (orchestrator), http://localhost:8000 (embedding)
```

## 🚀 Deployment Options

### Docker Compose (Recommended for Development)
```bash
# Standard deployment
docker-compose up -d

# With MongoDB Atlas
docker-compose -f docker-compose.atlas.yml up -d

# Production build
docker-compose -f docker-compose.prod.yml up -d
```

### AWS Deployment
Comprehensive AWS deployment guide available: [AWS Deployment Guide](docs/aws-deployment-guide.md)

#### Quick AWS Options:
- **AWS Lambda**: Serverless, cost-effective for variable workloads
- **Amazon ECS**: Containerized, consistent performance
- **EC2 Instances**: Full control, high performance
- **SageMaker**: ML-optimized for model-heavy workloads

```bash
# Example: Deploy to AWS Lambda
cd services/embedding_service
serverless deploy

cd services/orchestrator
serverless deploy
```

### Local Development
```bash
# Run services individually
cd services/embedding_service
python main.py

cd services/orchestrator
npm start
```

### Production Considerations
- Use environment-specific configuration files
- Set up proper logging and monitoring
- Configure SSL/TLS certificates
- Implement rate limiting and authentication
- Use managed database services (MongoDB Atlas recommended)

---

## 📚 API Specifications

> 📋 **Complete API Reference**: For detailed curl commands and examples, see [API_REFERENCE.md](docs/API_REFERENCE.md)

### Embedding Service (Port 8000)

#### 🔍 Health Check
```bash
GET /health
```
**Response:**
```json
{
  "status": "healthy",
  "service": "embedding_service",
  "version": "1.0.0"
}
```

#### 📄 Bulk Document Embedding (Used by ingestion)
```bash
POST /bulk_embed
Content-Type: application/json

{
  "documents": [
    {"id": "doc_1", "text": "Machine learning is a subset of AI..."},
    {"id": "doc_2", "text": "Deep learning uses neural networks..."}
  ]
}
```
**Response:**
```json
{
  "processed": 2,
  "results": [
    {
      "id": "doc_1",
      "embedding": [0.1, 0.2, ...],
      "dimension": 1024
    }
  ]
}
```

#### 🔎 Similarity Search (Used by RAG)
```bash
# Default cosine similarity
GET /search?query=What%20is%20machine%20learning&k=3

# Specify similarity metric
GET /search?query=What%20is%20machine%20learning&k=3&similarity_metric=cosine
GET /search?query=What%20is%20machine%20learning&k=3&similarity_metric=dot_product
```
**Response:**
```json
{
  "query": "What is machine learning",
  "results": [
    {
      "id": "ml_basics_1_chunk_0",
      "text": "Machine learning is a subset of AI that enables...",
      "score": 0.8542,
      "metadata": {}
    }
  ],
  "processing_time_ms": 245,
  "similarity_metric": "cosine"
}
```

### Orchestrator Service (Port 5000)

#### 🔍 Health Check
```bash
GET /health
```
**Response:**
```json
{
  "status": "healthy",
  "service": "rag-orchestrator",
  "version": "1.0.0",
  "timestamp": "2025-09-28T10:30:00.000Z",
  "dependencies": {
    "embedding_service": {
      "status": "healthy",
      "service": "embedding_service",
      "version": "1.0.0"
    }
  }
}
```

#### 💬 RAG Chat (Main endpoint)
```bash
POST /chat
Content-Type: application/json

{
  "user_id": "user_123",
  "query": "What is machine learning?",
  "k": 3,
  "similarity_metric": "cosine"
}
```

**Available Parameters:**
- `user_id`: Unique identifier for the user
- `query`: The question or prompt to process
- `k`: Number of similar documents to retrieve (default: 5)
- `similarity_metric`: Either "cosine" or "dot_product" (default: "cosine")
**Response:**
```json
{
  "answer": "Machine learning is a subset of AI that allows computers to learn and make decisions from data without being explicitly programmed...",
  "source_docs": [
    {
      "id": "ml_basics_1_chunk_0",
      "text": "Machine learning is a subset of AI that enables computers to learn...",
      "score": 0.8542,
      "metadata": {}
    }
  ],
  "timing_ms": 4250,
  "user_id": "user_123",
  "query": "What is machine learning?",
  "model_info": {
    "model": "gpt-3.5-turbo",
    "usage": {
      "prompt_tokens": 310,
      "completion_tokens": 99,
      "total_tokens": 409
    }
  },
  "timestamp": "2025-09-28T10:30:15.000Z"
}

## 🚀 Quick Commands Reference

### Essential Curl Commands (Copy & Paste Ready)

```bash
# Start services
docker-compose up -d

# Health checks
curl http://localhost:8000/health  # Embedding service
curl http://localhost:5000/health  # Orchestrator

# Ingest data
python scripts/ingest_data.py data/sample_documents.json --type json

# RAG Chat (Cosine similarity)
curl -X POST "http://localhost:5000/chat" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test_user", "query": "What is machine learning?", "k": 3, "similarity_metric": "cosine"}'

# RAG Chat (Dot product similarity)  
curl -X POST "http://localhost:5000/chat" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test_user", "query": "What is machine learning?", "k": 3, "similarity_metric": "dot_product"}'

# Direct vector search
curl "http://localhost:8000/search?query=machine%20learning&k=3&similarity_metric=cosine"
```

> 📋 **Full API Documentation**: [docs/API_REFERENCE.md](docs/API_REFERENCE.md) - Complete curl commands, parameters, and response formats

## 🎬 Demo Script

### Complete Workflow Demo

#### 1. Ingest Documents
```bash
# Install Python dependencies
pip install -r scripts/requirements.txt

# Ingest sample documents (20 AI/ML documents)
python scripts/ingest_data.py data/sample_documents.json --chunk-size 500 --chunk-overlap 50
```
**Expected Output:**
```
INFO - Starting ingestion process...
INFO - Processing 20 documents
INFO - Successfully processed document: ml_basics_1
INFO - Bulk embedding successful: 42 chunks processed
INFO - ✅ Ingestion completed: 42 chunks from 20 documents
```

#### 2. Query RAG System
```bash
# Ask a context-aware question (default cosine similarity)
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "demo_user",
    "query": "What is machine learning and how does it work?",
    "k": 3
  }'

# Or specify similarity metric explicitly
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "demo_user",
    "query": "What is machine learning and how does it work?",
    "k": 3,
    "similarity_metric": "dot_product"
  }'
```
**Expected Response:**
```json
{
  "answer": "Machine learning is a subset of AI that allows computers to learn and make decisions from data without being explicitly programmed for every scenario. It involves using algorithms to identify patterns in data and make predictions or decisions based on those patterns. The three main types of machine learning are supervised learning, unsupervised learning, and reinforcement learning...",
  "source_docs": [
    {
      "id": "ml_basics_1_chunk_0", 
      "text": "Machine learning is a subset of AI that enables computers to learn...",
      "score": 0.8542
    }
  ],
  "timing_ms": 4250,
  "model_info": {
    "model": "gpt-3.5-turbo",
    "usage": {"total_tokens": 409}
  }
}
```

#### 3. Train and Use LoRA Adapter
```bash
# Train LoRA adapter on toy QA data
python scripts/train_adapter.py
```
**Expected Output:**
```
🚀 LoRA Adapter Training Demo for Toy QA Data
📚 PHASE 1: Training LoRA Adapter
INFO - Starting LoRA training with distilgpt2
trainable params: 294,912 || all params: 82,575,616 || trainable%: 0.3571
INFO - Starting LoRA adapter training...
Training: 100%|████████| 8/8 [02:15<00:00]
INFO - ✅ LoRA adapter training completed and saved to ./toy_qa_lora_model

🤖 PHASE 2: Loading Adapter & Inference
--- Test 1 ---
Input: Question: What is Python?
Answer:
Generated: Python is a high-level programming language known for its simplicity and readability.
```

#### 4. Test Multi-Vector Similarity
```bash
# Quick demo of cosine vs dot product similarity
python scripts/quick_multi_vector_demo.py

# Comprehensive multi-vector similarity testing
python scripts/test_multi_vector.py
```

**Expected Output (Quick Demo):**
```
🔬 Multi-Vector Similarity Demo
==================================================
Query: What is machine learning?

📊 Testing Cosine Similarity...
✅ Success! (1.23s)
   Sources found: 3
   Score #1: 0.8542

⚡ Testing Dot Product Similarity...
✅ Success! (1.18s)
   Sources found: 3
   Score #1: 12.4837

📈 Comparison Summary:
   Cosine similarity timing: 1.23s
   Dot product timing: 1.18s
   Performance difference: 0.05s
```

#### 5. Interactive LoRA Usage
```bash
# Use trained LoRA model interactively
python scripts/use_lora_model.py
```
**Expected Session:**
```
🧠 LoRA QA Bot - Interactive Question Answering
============================================================
💭 Your Question: What is artificial intelligence?
🤖 Thinking...
📝 Answer: Artificial Intelligence is the simulation of human intelligence in machines.
--------------------------------------------------
💭 Your Question: quit
👋 Goodbye! Thanks for using LoRA QA Bot!
```

## ⚖️ Tradeoffs, Scaling & Cost Considerations

### Architecture Tradeoffs

| Component | Choice | Tradeoff | Alternative |
|-----------|--------|----------|-------------|
| **Embeddings** | Jina AI API | Cost vs. simplicity | Local sentence-transformers (slower, more memory) |
| **Vector DB** | MongoDB Atlas | Vendor lock-in vs. managed service | Self-hosted Chroma/Weaviate (more ops overhead) |
| **LLM** | OpenAI API | Cost vs. quality | Local Llama models (GPU required, slower) |
| **Orchestrator** | Node.js Express | Simple vs. feature-rich | Python FastAPI (more ML ecosystem) |
| **LoRA Training** | CPU-only | Slow training vs. accessibility | GPU training (faster, requires hardware) |

### Scaling Considerations

#### Performance Bottlenecks
1. **Embedding Generation**: 200-500ms per request to Jina AI
   - *Solution*: Batch requests, implement caching
2. **Vector Search**: O(n) similarity search in MongoDB
   - *Solution*: Optimize indexes, consider approximate search
3. **LLM Generation**: 2-5 seconds per response
   - *Solution*: Streaming responses, model caching

#### Horizontal Scaling
```yaml
# Scale embedding service for high throughput
embedding-service:
  deploy:
    replicas: 3
  
# Load balance orchestrator for concurrent users  
orchestrator:
  deploy:
    replicas: 2
```

#### Database Scaling
- **Read Replicas**: For search-heavy workloads
- **Sharding**: For >100M documents
- **Index Optimization**: Vector indexes consume significant RAM

### Cost Analysis (Monthly estimates)

| Component | Low Usage | Medium Usage | High Usage |
|-----------|-----------|--------------|------------|
| **Jina AI** (per 1K embeddings) | $0.10 | $25 | $200 |
| **OpenAI** (per 1K requests) | $2-5 | $50-100 | $500-1000 |
| **MongoDB Atlas** | $9 (M10) | $57 (M30) | $200+ (M40+) |
| **Hosting** (AWS/GCP) | $20-50 | $100-300 | $500+ |
| **Total** | $30-75 | $230-480 | $1400+ |

### Latency Optimization

#### Current Performance
- **Document Ingestion**: ~500ms per document (embedding + storage)
- **RAG Query**: ~4-6 seconds (search: 200ms, LLM: 4s)
- **LoRA Inference**: ~2-3 seconds (CPU-only)

#### Optimization Strategies
1. **Caching**: Redis for frequent queries (reduces 80% of LLM calls)
2. **Batching**: Process multiple embeddings together
3. **Prefetching**: Pre-compute embeddings for likely queries
4. **Model Optimization**: Use faster models (GPT-3.5-turbo vs GPT-4)
5. **Async Processing**: Non-blocking operations where possible

## 🧪 Testing

### Unit Tests

#### Python (Embedding Service)
```bash
# Run Python unit tests
cd services/embedding_service
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html
```

#### Node.js (Orchestrator)
```bash
# Run Node.js unit tests
cd services/orchestrator
npm test

# Run with coverage
npm run test:coverage
```

### Integration Tests

#### Multi-Vector Similarity Testing

**Quick Demo Script (`quick_multi_vector_demo.py`):**
```bash
# Start services first
docker-compose up -d

# Run quick demo (shows basic differences)
python scripts/quick_multi_vector_demo.py
```

**Features:**
- Simple comparison between cosine and dot product similarity
- Performance timing analysis
- Score comparison with explanations
- Direct embedding service testing
- User-friendly output with emojis and clear explanations

**Comprehensive Testing Script (`test_multi_vector.py`):**
```bash
# Run detailed multi-vector similarity comparison
python scripts/test_multi_vector.py
```

**Features:**
- Async testing with multiple queries
- Statistical analysis of score differences
- Performance benchmarking
- Detailed JSON results output
- Both RAG pipeline and direct embedding service testing

**Testing Scripts Output:**
- Performance comparison between cosine and dot product similarity
- Score distribution analysis
- Timing benchmarks with millisecond precision
- Detailed JSON results with statistical analysis
- Direct embedding service vs. full RAG pipeline comparison

#### React Demo UI Testing
```bash
# Start services
docker-compose up -d

# Install and run React demo
cd demo-ui
npm install
npm start

# Test features:
# - Chat functionality
# - Similarity metric selection
# - Source document display
# - Performance metrics
```

### CI/CD Testing

The GitHub Actions pipeline automatically runs:
- Python linting (flake8)
- Python unit tests (pytest)
- Node.js linting (eslint)
- Node.js unit tests (Jest)
- Docker image builds
- Security scanning (Trivy)

```yaml
# Trigger pipeline
git push origin main

# Or run specific workflow
gh workflow run ci.yml
```

### Performance Testing

```bash
# Load test the chat endpoint with cosine similarity
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user",
    "query": "What is machine learning?",
    "k": 5,
    "similarity_metric": "cosine"
  }'

# Test with dot product similarity
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user",
    "query": "What is machine learning?",
    "k": 5,
    "similarity_metric": "dot_product"
  }'

# Run comprehensive multi-vector performance tests
python scripts/test_multi_vector.py
```

### Monitoring Tests

```bash
# Check Prometheus metrics
curl http://localhost:8000/metrics  # Embedding service
curl http://localhost:5000/metrics  # Orchestrator

# Verify metric collection
grep "http_requests_total" logs/combined.log
```

---

**System Status**: ✅ Fully functional RAG system with LoRA fine-tuning capabilities

## ✨ Key Features

- **🔧 LoRA Fine-tuning**: Complete PEFT adapter training with distilGPT-2, toy QA dataset, and interactive inference
- **🧠 RAG System**: Complete Retrieval-Augmented Generation pipeline with vector search
- **🔄 Microservices Architecture**: FastAPI embedding service and Node.js orchestrator
- **📊 Vector Database**: MongoDB Atlas with vector search capabilities
- **📏 Multi-Vector Similarity**: Support for both cosine similarity and dot product similarity metrics
- **🐳 Containerized Deployment**: Docker Compose setup for easy deployment
- **🚀 CI/CD Pipeline**: GitHub Actions workflow with automated testing and Docker builds
- **📈 Monitoring & Metrics**: Prometheus metrics integration for both services
- **🎨 React Demo UI**: Interactive web interface for testing the RAG system
- **🧪 Comprehensive Testing**: Unit tests, multi-vector similarity tests, and performance benchmarks
- **🔑 API Integration**: OpenAI API for embeddings and chat completions
- **☁️ AWS Deployment Guide**: Complete migration guide for AWS Lambda, ECS, EC2, and SageMaker

## 🆕 New Advanced Features

### 1. **Multi-Vector Similarity Support**
- Configurable similarity metrics: cosine similarity and dot product
- API parameter: `similarity_metric` in chat requests
- Performance comparison tools included

### 2. **CI/CD Pipeline**
- GitHub Actions workflow with multi-stage pipeline
- Automated linting, testing, and Docker image building
- Security scanning with Trivy
- Automated deployment capabilities

### 3. **Monitoring & Metrics**
- Prometheus metrics for both Python and Node.js services
- Request latency, error counts, and custom metrics
- `/metrics` endpoints for both services
- Ready for Grafana dashboard integration

### 4. **React Demo UI**
- Modern, responsive web interface
- Real-time chat functionality
- Source document display with scores
- Similarity metric selection
- Performance timing display

### 5. **Comprehensive Testing**
- Python unit tests with pytest
- Node.js tests with Jest
- Multi-vector similarity testing script
- Automated test execution in CI/CD
