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
│   ├── lora_qa_demo.py          # Complete LoRA training & inference demo
│   ├── use_lora_model.py        # Interactive LoRA model usage script
│   ├── ingest_data.py           # Data ingestion pipeline
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
JINA_API_KEY=your_jina_api_key_here          # Get from https://jina.ai
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
curl http://localhost:3000/health  # Orchestrator

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

## 📚 API Specifications

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
GET /search?query=What%20is%20machine%20learning&k=3
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
  "processing_time_ms": 245
}
```

### Orchestrator Service (Port 3000)

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
  "k": 3
}
```
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
# Ask a context-aware question
curl -X POST http://localhost:3000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "demo_user",
    "query": "What is machine learning and how does it work?",
    "k": 3
  }'
```
**Expected Response:**
```json
{
  "answer": "Machine learning is a subset of AI that allows computers to learn and make decisions from data without being explicitly programmed for every scenario. It involves using algorithms to identify patterns in data and make predictions or decisions based on those patterns. The three main types of machine learning are supervised learning, unsupervised learning, and reinforcement learning...",
  "source_docs": [
    {
      "id": "ml_basics_1_chunk_0", 
      "text": "Machine learning is a subset of AI that enables computers...",
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
python scripts/lora_qa_demo.py
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

#### 4. Interactive LoRA Usage
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

## 🧪 Testing the System

### Health Checks
```bash
# Check all services are running
curl http://localhost:8000/health  # Embedding service
curl http://localhost:3000/health  # Orchestrator

# Expected responses: {"status": "healthy", ...}
```

### Manual API Testing
```bash
# Test document search directly
curl "http://localhost:8000/search?query=artificial%20intelligence&k=2"

# Test RAG with different questions
curl -X POST http://localhost:3000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test", "query": "How does deep learning work?", "k": 2}'
```

### Performance Testing
```bash
# Monitor response times
time curl -X POST http://localhost:3000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test", "query": "What is supervised learning?", "k": 3}'

# Expected: ~4-6 seconds total response time
```

## 🔧 Configuration

### Required Environment Variables
```bash
# API Keys (required)
JINA_API_KEY=your_jina_api_key_here          # Get from https://jina.ai
OPENAI_API_KEY=your_openai_api_key_here      # Get from https://openai.com

# MongoDB Atlas (required)
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/
MONGODB_DATABASE=rag_system
MONGODB_COLLECTION=embeddings

# Optional settings (have defaults)
LLM_MODEL=gpt-3.5-turbo
CHUNK_SIZE=500
CHUNK_OVERLAP=50
```

### Data Processing Settings
- **Chunk Size**: 500 characters per document chunk
- **Chunk Overlap**: 50 characters (10% overlap for context continuity)
- **Vector Dimensions**: 1024 (Jina AI embeddings)
- **Similarity**: Cosine similarity for vector search

## 📁 Project Structure

```
ai-engineer-challenge/
├── services/
│   ├── embedding_service/          # Python FastAPI service
│   │   ├── main.py                # FastAPI app with /health, /bulk_embed, /search
│   │   ├── models.py              # Pydantic request/response models  
│   │   ├── embedding_service.py   # Jina AI integration
│   │   ├── vector_store.py        # MongoDB Atlas vector operations
│   │   └── Dockerfile            # Container config
│   └── orchestrator/              # Node.js Express service  
│       ├── src/
│       │   ├── app.js            # Express app with /health, /chat
│       │   ├── embeddingClient.js # Calls embedding service
│       │   └── llmService.js     # OpenAI API integration
│       └── Dockerfile           # Container config
├── scripts/
│   ├── lora_qa_demo.py          # Complete LoRA training & inference 
│   ├── use_lora_model.py        # Interactive LoRA usage
│   ├── ingest_data.py           # Document ingestion pipeline
│   └── toy_qa_lora_model/       # Trained LoRA adapter (1.6MB)
├── data/
│   └── sample_documents.json     # 20 AI/ML documents for demo
├── docker-compose.atlas.yml      # Multi-service orchestration
└── README.md                     # This file
```

## � Common Issues & Solutions

### LoRA Training Issues
```bash
# If model not found:
python scripts/lora_qa_demo.py  # Train model first

# If memory issues:
# Reduce batch_size in lora_qa_demo.py from 2 to 1
```

### RAG System Issues  
```bash
# If services won't start:
docker-compose -f docker-compose.atlas.yml down
docker-compose -f docker-compose.atlas.yml up -d

# If no search results:
# Check if documents were ingested successfully
python scripts/ingest_data.py data/sample_documents.json
```

---

**System Status**: ✅ Fully functional RAG system with LoRA fine-tuning capabilities

**Key Features**: Document ingestion, vector search, context-aware Q&A, LoRA adapter training, interactive model usage
