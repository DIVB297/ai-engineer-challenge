# API Reference - RAG System

Complete API documentation with copy-paste ready curl commands for the RAG system.

## 🏁 Quick Start

Start the services first:
```bash
docker-compose up -d
```

Check service health:
```bash
curl http://localhost:8000/health  # Embedding service
curl http://localhost:5000/health  # Orchestrator
```

---

## 🔧 Embedding Service API (Port 8000)

### Health Check
```bash
curl http://localhost:8000/health
```

### Single Document Embedding
```bash
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{
    "id": "test_doc", 
    "text": "This is a test document about artificial intelligence and machine learning."
  }'
```

### Bulk Document Embedding
```bash
curl -X POST "http://localhost:8000/bulk_embed" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {
        "id": "test_doc_1",
        "text": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed."
      },
      {
        "id": "test_doc_2", 
        "text": "Deep learning uses neural networks with multiple layers to model and understand complex patterns in data, making it powerful for tasks like image recognition and natural language processing."
      },
      {
        "id": "test_doc_3",
        "text": "Natural language processing (NLP) is a field of AI that focuses on the interaction between computers and human language, enabling machines to understand, interpret, and generate human text."
      }
    ]
  }'
```

### Vector Search - Cosine Similarity (Default)
```bash
curl "http://localhost:8000/search?query=machine%20learning%20algorithms&k=3"
```

### Vector Search - Dot Product Similarity
```bash
curl "http://localhost:8000/search?query=machine%20learning%20algorithms&k=3&similarity_metric=dot_product"
```

### Vector Search - with URL-encoded Query
```bash
curl "http://localhost:8000/search?query=What%20is%20artificial%20intelligence&k=5&similarity_metric=cosine"
```

---

## 🎯 Orchestrator API (Port 5000)

### Health Check
```bash
curl http://localhost:5000/health
```

### RAG Chat - Basic Query
```bash
curl -X POST "http://localhost:5000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user_123",
    "query": "What is machine learning and how does it work?",
    "k": 3
  }'
```

### RAG Chat - with Cosine Similarity
```bash
curl -X POST "http://localhost:5000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user_123",
    "query": "What is machine learning and how does it work?",
    "k": 3,
    "similarity_metric": "cosine"
  }'
```

### RAG Chat - with Dot Product Similarity
```bash
curl -X POST "http://localhost:5000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user_123",
    "query": "Explain deep learning and neural networks",
    "k": 5,
    "similarity_metric": "dot_product"
  }'
```

### RAG Chat - Complex Query
```bash
curl -X POST "http://localhost:5000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "ai_researcher_456",
    "query": "What are the differences between supervised and unsupervised learning? Provide examples of each.",
    "k": 4,
    "similarity_metric": "cosine"
  }'
```

---

## 📊 Data Ingestion Commands

### Ingest JSON Documents
```bash
# From project root directory
python scripts/ingest_data.py data/sample_documents.json --type json
```

### Ingest Text File (Auto-detect)
```bash
# From project root directory  
python scripts/ingest_data.py data/rag_explanation.txt --type auto
```

### Ingest with Virtual Environment
```bash
# Using virtual environment Python
/Users/macair/Divansh/Project/ai-engineer-challenge/venv/bin/python scripts/ingest_data.py data/sample_documents.json --type json

/Users/macair/Divansh/Project/ai-engineer-challenge/venv/bin/python scripts/ingest_data.py data/rag_explanation.txt --type auto
```

### Ingest with Custom Chunking
```bash
python scripts/ingest_data.py data/sample_documents.json \
  --type json \
  --chunk-size 500 \
  --chunk-overlap 50
```

---

## 🧪 Testing & Demo Commands

### Quick Multi-Vector Demo
```bash
python scripts/quick_multi_vector_demo.py
```

### Comprehensive Multi-Vector Testing
```bash
python scripts/test_multi_vector.py
```

### LoRA Training & Testing
```bash
# Train LoRA adapter
python scripts/train_adapter.py

# Interactive LoRA usage
python scripts/use_lora_model.py
```

---

## 📈 Monitoring & Metrics

### Prometheus Metrics
```bash
# Embedding service metrics
curl http://localhost:8000/metrics

# Orchestrator metrics
curl http://localhost:5000/metrics
```

---

## 🔄 Workflow Examples

### Complete RAG Workflow
```bash
# 1. Start services
docker-compose up -d

# 2. Ingest data
python scripts/ingest_data.py data/sample_documents.json --type json

# 3. Query with cosine similarity
curl -X POST "http://localhost:5000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "demo_user",
    "query": "What is machine learning?",
    "k": 3,
    "similarity_metric": "cosine"
  }'

# 4. Query with dot product similarity
curl -X POST "http://localhost:5000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "demo_user",
    "query": "What is machine learning?",
    "k": 3,
    "similarity_metric": "dot_product"
  }'
```

### Comparison Testing Workflow
```bash
# 1. Test embedding service directly
curl "http://localhost:8000/search?query=artificial%20intelligence&k=3&similarity_metric=cosine"
curl "http://localhost:8000/search?query=artificial%20intelligence&k=3&similarity_metric=dot_product"

# 2. Test full RAG pipeline
curl -X POST "http://localhost:5000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "comparison_test",
    "query": "artificial intelligence",
    "k": 3,
    "similarity_metric": "cosine"
  }'

curl -X POST "http://localhost:5000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "comparison_test",
    "query": "artificial intelligence", 
    "k": 3,
    "similarity_metric": "dot_product"
  }'
```

---

## 📋 Expected Response Formats

### Embedding Service Search Response
```json
{
  "query": "machine learning algorithms",
  "results": [
    {
      "id": "ml_basics_1_chunk_0",
      "text": "Machine learning algorithms are computational methods...",
      "score": 0.8542,
      "metadata": {}
    }
  ],
  "processing_time_ms": 245,
  "similarity_metric": "cosine"
}
```

### Orchestrator Chat Response
```json
{
  "answer": "Machine learning is a subset of artificial intelligence...",
  "source_docs": [
    {
      "id": "ml_basics_1_chunk_0",
      "text": "Machine learning is a subset of AI...",
      "score": 0.8542,
      "metadata": {}
    }
  ],
  "timing_ms": 4250,
  "user_id": "test_user_123",
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
```

---

## 🚨 Error Handling Examples

### Missing Parameters
```bash
# This will return 400 Bad Request
curl -X POST "http://localhost:5000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?"
  }'
```

### Invalid Similarity Metric
```bash
# This will use default cosine similarity
curl "http://localhost:8000/search?query=test&similarity_metric=invalid"
```

---

## 🔧 Troubleshooting Commands

### Check Service Status
```bash
docker-compose ps
docker-compose logs embedding-service
docker-compose logs orchestrator
```

### Test Connectivity
```bash
# Test if services are responding
curl -f http://localhost:8000/health || echo "Embedding service down"
curl -f http://localhost:5000/health || echo "Orchestrator down"
```

### Environment Variables Check
```bash
# Check if OpenAI API key is set in orchestrator
docker-compose exec orchestrator env | grep OPENAI

# Check MongoDB connection in embedding service
docker-compose exec embedding-service env | grep MONGODB
```

---

## 📝 Notes

- Replace `localhost` with your server IP/domain for remote access
- All timestamps are in ISO 8601 format
- Scores range from 0-1 for cosine similarity, unbounded for dot product
- The `k` parameter controls number of similar documents returned (default: 5)
- User IDs should be unique strings for tracking purposes
- Query strings in URLs should be URL-encoded for special characters

---

**Quick Reference**: All services must be running before executing these commands. Use `docker-compose up -d` to start all services.
