from pydantic import BaseModel
from typing import List, Optional

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
