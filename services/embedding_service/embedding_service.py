import os
import httpx
import asyncio
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

class OpenAIEmbeddingService:
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.openai.com/v1/embeddings"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text using OpenAI API"""
        async with httpx.AsyncClient() as client:
            payload = {
                "model": self.model,
                "input": text
            }
            
            response = await client.post(
                self.base_url,
                json=payload,
                headers=self.headers,
                timeout=30.0
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["data"][0]["embedding"]
            else:
                raise Exception(f"OpenAI API error: {response.status_code} - {response.text}")
    
    async def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts using OpenAI API"""
        async with httpx.AsyncClient() as client:
            payload = {
                "model": self.model,
                "input": texts
            }
            
            response = await client.post(
                self.base_url,
                json=payload,
                headers=self.headers,
                timeout=60.0
            )
            
            if response.status_code == 200:
                result = response.json()
                return [item["embedding"] for item in result["data"]]
            else:
                raise Exception(f"OpenAI API error: {response.status_code} - {response.text}")

class LocalEmbeddingService:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        
    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text using local model"""
        embedding = self.model.encode([text])[0]
        return embedding.tolist()
    
    async def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embedings for multiple texts using local model"""
        embeddings = self.model.encode(texts)
        return [emb.tolist() for emb in embeddings]

class EmbeddingServiceFactory:
    @staticmethod
    def create_service(use_jina: bool = True, api_key: str = None, openai_api_key: str = None):
        if openai_api_key:
            return OpenAIEmbeddingService(openai_api_key)
        else:
            return LocalEmbeddingService()
