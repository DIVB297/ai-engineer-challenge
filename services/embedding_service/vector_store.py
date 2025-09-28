import os
import asyncio
import time
import numpy as np
from typing import List
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import IndexModel, TEXT
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, connection_string: str, database_name: str, collection_name: str):
        # Configure client for both Atlas and local MongoDB
        if "mongodb.net" in connection_string or "mongodb+srv" in connection_string:
            # Atlas connection with SSL configuration that handles certificate issues
            self.client = AsyncIOMotorClient(
                connection_string,
                tls=True,
                tlsAllowInvalidCertificates=True,
                serverSelectionTimeoutMS=30000,
                connectTimeoutMS=20000,
                socketTimeoutMS=20000
            )
            self.is_atlas = True
        else:
            # Local MongoDB
            self.client = AsyncIOMotorClient(
                connection_string,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000
            )
            self.is_atlas = False
            
        self.database = self.client[database_name]
        self.collection = self.database[collection_name]
        self.vector_search_available = False
        self.embedding_dimensions = None
        
    async def initialize(self):
        """Initialize the vector store with proper indexes"""
        try:
            # Always create basic indexes
            await self.collection.create_index("id", unique=True)
            text_index = IndexModel([("text", TEXT), ("id", 1)])
            await self.collection.create_indexes([text_index])
            
            # For Atlas, we'll create the vector index when we get the first embedding
            # This allows us to detect the correct dimensions automatically
            if self.is_atlas:
                logger.info("Atlas detected - vector search index will be created on first embedding")
                self.vector_search_available = False  # Will be set to True after first embedding
            else:
                logger.info("Using local MongoDB - vector search not available")
                self.vector_search_available = False
                
            logger.info("Vector store initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise
    
    async def _ensure_vector_index(self, embedding_dimensions: int):
        """Ensure vector search index exists with correct dimensions"""
        if not self.is_atlas or self.vector_search_available:
            return
            
        try:
            # Check if we need to create or update the index
            if self.embedding_dimensions is None:
                self.embedding_dimensions = embedding_dimensions
                
            if self.embedding_dimensions != embedding_dimensions:
                logger.warning(f"Embedding dimensions changed from {self.embedding_dimensions} to {embedding_dimensions}")
                self.embedding_dimensions = embedding_dimensions
                
            # Create vector search index with correct dimensions
            await self.collection.create_search_index({
                "definition": {
                    "mappings": {
                        "dynamic": True,
                        "fields": {
                            "embedding": {
                                "type": "knnVector",
                                "dimensions": embedding_dimensions,
                                "similarity": "cosine"
                            }
                        }
                    }
                },
                "name": "vector_index"
            })
            self.vector_search_available = True
            logger.info(f"MongoDB Atlas Vector Search initialized with {embedding_dimensions} dimensions")
        except Exception as vector_error:
            logger.warning(f"Vector search index creation failed (will use fallback): {vector_error}")
            self.vector_search_available = False
            
    async def upsert_embedding(self, doc_id: str, text: str, embedding: List[float], metadata: dict = None):
        """Insert or update a single document with its embedding"""
        try:
            # Ensure vector index exists with correct dimensions
            await self._ensure_vector_index(len(embedding))
            
            document = {
                "id": doc_id,
                "text": text,
                "embedding": embedding,
                "metadata": metadata or {},
                "created_at": time.time()
            }
            
            result = await self.collection.replace_one(
                {"id": doc_id},
                document,
                upsert=True
            )
            
            return result.upserted_id or result.modified_count > 0
        except Exception as e:
            logger.error(f"Error upserting document {doc_id}: {e}")
            raise
    
    async def upsert_embeddings_batch(self, documents: List[dict]):
        """Insert or update multiple documents with their embeddings"""
        try:
            if not documents:
                return 0
                
            # Ensure vector index exists with correct dimensions (use first document's embedding)
            first_embedding = documents[0].get("embedding", [])
            if first_embedding:
                await self._ensure_vector_index(len(first_embedding))
            
            # Use individual replace operations for better compatibility
            success_count = 0
            for doc in documents:
                try:
                    logger.info(f"Processing document {doc['id']} for batch upsert")
                    document = {
                        "id": doc["id"],
                        "text": doc["text"],
                        "embedding": doc["embedding"],
                        "metadata": doc.get("metadata", {}),
                        "created_at": time.time()
                    }
                    
                    logger.info(f"Calling replace_one for document {doc['id']}")
                    result = await self.collection.replace_one(
                        {"id": doc["id"]},
                        document,
                        upsert=True
                    )
                    logger.info(f"Replace_one completed for document {doc['id']}, result: {result}")
                    
                    if result.upserted_id or result.modified_count > 0:
                        success_count += 1
                        logger.info(f"Successfully processed document {doc['id']}")
                except Exception as doc_error:
                    logger.error(f"Error upserting document {doc['id']}: {doc_error}")
                    raise
                    
            logger.info(f"Batch upsert completed, success_count: {success_count}")
            return success_count
        except Exception as e:
            logger.error(f"Error batch upserting documents: {e}")
            raise
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

    def _convert_objectid_to_string(self, doc):
        """Convert MongoDB ObjectId to string for JSON serialization"""
        if "_id" in doc and hasattr(doc["_id"], "str"):
            doc["_id"] = str(doc["_id"])
        return doc

    async def similarity_search(self, query_embedding, k=5, similarity_metric="cosine"):
        """Perform similarity search using MongoDB Atlas Vector Search"""
        try:
            # Configure search based on similarity metric
            if similarity_metric == "cosine":
                # Use default Atlas vector search (cosine similarity)
                pipeline = [
                    {
                        "$vectorSearch": {
                            "index": "vector_index",
                            "path": "embedding",
                            "queryVector": query_embedding,
                            "numCandidates": k * 10,
                            "limit": k
                        }
                    },
                    {
                        "$project": {
                            "id": 1,
                            "text": 1,
                            "metadata": 1,
                            "score": {"$meta": "vectorSearchScore"}
                        }
                    }
                ]
            elif similarity_metric == "dot_product":
                # Use aggregation pipeline for dot product similarity
                pipeline = [
                    {
                        "$addFields": {
                            "score": {
                                "$reduce": {
                                    "input": {"$zip": {"inputs": ["$embedding", query_embedding]}},
                                    "initialValue": 0,
                                    "in": {"$add": ["$$value", {"$multiply": [{"$arrayElemAt": ["$$this", 0]}, {"$arrayElemAt": ["$$this", 1]}]}]}
                                }
                            }
                        }
                    },
                    {
                        "$sort": {"score": -1}
                    },
                    {
                        "$limit": k
                    },
                    {
                        "$project": {
                            "id": 1,
                            "text": 1,
                            "metadata": 1,
                            "score": 1
                        }
                    }
                ]
            else:
                raise ValueError(f"Unsupported similarity metric: {similarity_metric}")

            cursor = self.collection.aggregate(pipeline)
            results = []
            
            async for doc in cursor:
                results.append({
                    "id": doc["id"],
                    "text": doc["text"],
                    "score": doc["score"],
                    "metadata": doc.get("metadata", {}),
                    "similarity_metric": similarity_metric
                })

            logger.info(f"Vector search completed with {similarity_metric}, found {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            raise
    
    async def delete_document(self, doc_id: str):
        """Delete a document by ID"""
        try:
            result = await self.collection.delete_one({"id": doc_id})
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}")
            raise
    
    async def delete_stale_documents(self, cutoff_time: float):
        """Delete documents older than cutoff_time"""
        try:
            result = await self.collection.delete_many({"created_at": {"$lt": cutoff_time}})
            logger.info(f"Deleted {result.deleted_count} stale documents")
            return result.deleted_count
        except Exception as e:
            logger.error(f"Error deleting stale documents: {e}")
            raise
    
    async def get_document_count(self):
        """Get total number of documents in the collection"""
        try:
            return await self.collection.count_documents({})
        except Exception as e:
            logger.error(f"Error getting document count: {e}")
            raise
    
    async def close(self):
        """Close the database connection"""
        self.client.close()
