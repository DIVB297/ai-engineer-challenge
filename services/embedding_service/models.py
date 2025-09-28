from typing import List

from pydantic import BaseModel


class EmbedRequest(BaseModel):
    id: str
    text: str


class BulkEmbedRequest(BaseModel):
    documents: List[EmbedRequest]


class EmbedResponse(BaseModel):
    id: str
    embedding: List[float]
    dimension: int


class BulkEmbedResponse(BaseModel):
    processed: int
    results: List[EmbedResponse]


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    similarity_metric: str = "cosine"  # "cosine" or "dot_product"


class SearchResult(BaseModel):
    id: str
    text: str
    score: float
    metadata: dict = {}
    similarity_metric: str


class SearchResponse(BaseModel):
    results: List[SearchResult]
