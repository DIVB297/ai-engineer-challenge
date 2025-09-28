import os
import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from models import EmbedRequest, BulkEmbedRequest, EmbedResponse, BulkEmbedResponse, HealthResponse
from embedding_service import EmbeddingServiceFactory
from vector_store import VectorStore

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
embedding_service = None
vector_store = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global embedding_service, vector_store
    
    # Initialize embedding service
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    embedding_service = EmbeddingServiceFactory.create_service(
        openai_api_key=openai_api_key
    )
    
    # Initialize vector store
    mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    mongodb_database = os.getenv("MONGODB_DATABASE", "rag_system")
    mongodb_collection = os.getenv("MONGODB_COLLECTION", "embeddings")
    
    vector_store = VectorStore(mongodb_uri, mongodb_database, mongodb_collection)
    
    try:
        await vector_store.initialize()
        logger.info("Services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    
    yield
    
    # Shutdown
    if vector_store:
        await vector_store.close()
    logger.info("Services shut down successfully")

# Create FastAPI app
app = FastAPI(
    title="RAG Embedding Service",
    description="Embedding service for RAG system using MongoDB Atlas Vector Search with cosine similarity",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_embedding_service():
    return embedding_service

def get_vector_store():
    return vector_store

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        service="embedding_service",
        version="1.0.0"
    )

@app.post("/embed", response_model=EmbedResponse)
async def embed_document(
    request: EmbedRequest,
    embed_service=Depends(get_embedding_service),
    store=Depends(get_vector_store)
):
    """Generate embedding for a single document and store it"""
    try:
        start_time = time.time()
        
        # Generate embedding
        embedding = await embed_service.get_embedding(request.text)
        
        # Store in vector database
        await store.upsert_embedding(
            doc_id=request.id,
            text=request.text,
            embedding=embedding
        )
        
        processing_time = time.time() - start_time
        logger.info(f"Processed document {request.id} in {processing_time:.3f}s")
        
        return EmbedResponse(
            id=request.id,
            embedding=embedding,
            dimension=len(embedding)
        )
        
    except Exception as e:
        logger.error(f"Error processing document {request.id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/bulk_embed", response_model=BulkEmbedResponse)
async def bulk_embed_documents(
    request: BulkEmbedRequest,
    embed_service=Depends(get_embedding_service),
    store=Depends(get_vector_store)
):
    """Generate embeddings for multiple documents and store them"""
    try:
        start_time = time.time()
        
        # Extract texts and IDs
        texts = [doc.text for doc in request.documents]
        doc_ids = [doc.id for doc in request.documents]
        
        # Generate embeddings in batch
        embeddings = await embed_service.get_embeddings_batch(texts)
        
        # Prepare documents for bulk insert
        documents = []
        results = []
        
        for i, (doc, embedding) in enumerate(zip(request.documents, embeddings)):
            documents.append({
                "id": doc.id,
                "text": doc.text,
                "embedding": embedding
            })
            
            results.append(EmbedResponse(
                id=doc.id,
                embedding=embedding,
                dimension=len(embedding)
            ))
        
        # Bulk insert into vector store
        try:
            processed_count = await store.upsert_embeddings_batch(documents)
        except Exception as store_error:
            logger.error(f"Error storing embeddings: {store_error}")
            raise HTTPException(status_code=500, detail="Failed to store embeddings in vector database")
        
        processing_time = time.time() - start_time
        logger.info(f"Bulk processed {len(request.documents)} documents in {processing_time:.3f}s")
        
        return BulkEmbedResponse(
            processed=processed_count,
            results=results
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in bulk processing: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during bulk processing")

@app.get("/search")
async def search_similar_documents(
    query: str,
    k: int = 5,
    embed_service=Depends(get_embedding_service),
    store=Depends(get_vector_store)
):
    """Search for similar documents using vector similarity"""
    try:
        start_time = time.time()
        
        # Generate query embedding
        query_embedding = await embed_service.get_embedding(query)
        
        # Perform similarity search
        results = await store.similarity_search(query_embedding, k=k)
        
        processing_time = time.time() - start_time
        logger.info(f"Search completed in {processing_time:.3f}s, found {len(results)} results")
        
        return {
            "query": query,
            "results": results,
            "processing_time_ms": processing_time * 1000
        }
        
    except Exception as e:
        logger.error(f"Error in search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("EMBEDDING_SERVICE_PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
