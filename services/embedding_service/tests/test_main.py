import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient

from main import app
from models import EmbedRequest, BulkEmbedRequest
from embedding_service import EmbeddingServiceFactory
from vector_store import VectorStore


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_embedding_service():
    service = Mock()
    service.get_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])
    service.get_embeddings_batch = AsyncMock(return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    return service


@pytest.fixture
def mock_vector_store():
    store = Mock()
    store.upsert_embedding = AsyncMock(return_value=True)
    store.upsert_embeddings_batch = AsyncMock(return_value=2)
    store.similarity_search = AsyncMock(return_value=[
        {"id": "test_1", "text": "Test document", "score": 0.95, "metadata": {}}
    ])
    return store


class TestHealthEndpoint:
    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "embedding_service"


class TestEmbeddingEndpoints:
    @patch('main.get_embedding_service')
    @patch('main.get_vector_store')
    def test_bulk_embed_success(self, mock_get_store, mock_get_service, client, 
                              mock_embedding_service, mock_vector_store):
        mock_get_service.return_value = mock_embedding_service
        mock_get_store.return_value = mock_vector_store
        
        request_data = {
            "documents": [
                {"id": "doc_1", "text": "Test document 1"},
                {"id": "doc_2", "text": "Test document 2"}
            ]
        }
        
        response = client.post("/bulk_embed", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["processed"] == 2
        assert len(data["results"]) == 2
        assert data["results"][0]["id"] == "doc_1"

    @patch('main.get_embedding_service')
    @patch('main.get_vector_store')
    def test_search_success(self, mock_get_store, mock_get_service, client,
                          mock_embedding_service, mock_vector_store):
        mock_get_service.return_value = mock_embedding_service
        mock_get_store.return_value = mock_vector_store
        
        response = client.get("/search?query=test&k=5")
        assert response.status_code == 200
        
        data = response.json()
        assert data["query"] == "test"
        assert len(data["results"]) == 1
        assert "processing_time_ms" in data


class TestEmbeddingService:
    @pytest.mark.asyncio
    async def test_jina_embedding_service(self):
        with patch('embedding_service.JinaEmbeddingService') as mock_jina:
            mock_instance = AsyncMock()
            mock_instance.get_embedding.return_value = [0.1, 0.2, 0.3]
            mock_jina.return_value = mock_instance
            
            service = EmbeddingServiceFactory.create_service(jina_api_key="test_key")
            result = await service.get_embedding("test text")
            
            assert result == [0.1, 0.2, 0.3]
            mock_instance.get_embedding.assert_called_once_with("test text")


class TestVectorStore:
    @pytest.mark.asyncio
    async def test_vector_store_initialization(self):
        with patch('vector_store.AsyncIOMotorClient') as mock_client:
            store = VectorStore("mongodb://test", "test_db", "test_collection")
            
            # Test initialization doesn't raise exception
            assert store.uri == "mongodb://test"
            assert store.database_name == "test_db"
            assert store.collection_name == "test_collection"


class TestModels:
    def test_embed_request_validation(self):
        # Valid request
        request = EmbedRequest(id="test_1", text="Test document")
        assert request.id == "test_1"
        assert request.text == "Test document"
        
        # Invalid request - missing text
        with pytest.raises(ValueError):
            EmbedRequest(id="test_1")

    def test_bulk_embed_request_validation(self):
        # Valid request
        request = BulkEmbedRequest(documents=[
            {"id": "doc_1", "text": "Document 1"},
            {"id": "doc_2", "text": "Document 2"}
        ])
        assert len(request.documents) == 2
        
        # Invalid request - empty documents
        with pytest.raises(ValueError):
            BulkEmbedRequest(documents=[])


if __name__ == "__main__":
    pytest.main([__file__])
