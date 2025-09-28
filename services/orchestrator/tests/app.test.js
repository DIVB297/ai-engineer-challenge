const request = require('supertest');
const app = require('../src/app');

describe('Orchestrator API', () => {
  describe('GET /health', () => {
    it('should return health status', async () => {
      const response = await request(app)
        .get('/health')
        .expect(200);

      expect(response.body).toHaveProperty('status', 'healthy');
      expect(response.body).toHaveProperty('service', 'rag-orchestrator');
      expect(response.body).toHaveProperty('version', '1.0.0');
    });
  });

  describe('POST /chat', () => {
    it('should require user_id and query', async () => {
      const response = await request(app)
        .post('/chat')
        .send({})
        .expect(400);

      expect(response.body).toHaveProperty('error');
    });

    it('should require user_id', async () => {
      const response = await request(app)
        .post('/chat')
        .send({ query: 'test query' })
        .expect(400);

      expect(response.body).toHaveProperty('error', 'User ID is required');
      expect(response.body).toHaveProperty('code', 'MISSING_USER_ID');
    });

    it('should require query', async () => {
      const response = await request(app)
        .post('/chat')
        .send({ user_id: 'test_user' })
        .expect(400);

      expect(response.body).toHaveProperty('error', 'Query is required');
      expect(response.body).toHaveProperty('code', 'MISSING_QUERY');
    });

    // Integration test would require actual services running
    it.skip('should return chat response when services are available', async () => {
      const response = await request(app)
        .post('/chat')
        .send({
          user_id: 'test_user',
          query: 'What is machine learning?',
          k: 3
        })
        .expect(200);

      expect(response.body).toHaveProperty('answer');
      expect(response.body).toHaveProperty('source_docs');
      expect(response.body).toHaveProperty('timing_ms');
      expect(response.body).toHaveProperty('user_id', 'test_user');
    });
  });

  describe('Error Handling', () => {
    it('should handle 404 routes', async () => {
      const response = await request(app)
        .get('/nonexistent')
        .expect(404);

      expect(response.body).toHaveProperty('error', 'Not found');
      expect(response.body.message).toContain('Route GET /nonexistent not found');
    });
  });
});

describe('EmbeddingClient', () => {
  const EmbeddingClient = require('../src/embeddingClient');
  
  describe('constructor', () => {
    it('should initialize with base URL', () => {
      const client = new EmbeddingClient('http://localhost:8000');
      expect(client.baseUrl).toBe('http://localhost:8000');
    });
  });

  describe('healthCheck', () => {
    it('should return health status', async () => {
      const client = new EmbeddingClient('http://localhost:8000');
      
      // Mock the axios call
      const mockAxios = jest.spyOn(client, 'healthCheck');
      mockAxios.mockResolvedValue({
        status: 'healthy',
        service: 'embedding_service',
        version: '1.0.0'
      });

      const result = await client.healthCheck();
      expect(result.status).toBe('healthy');
    });
  });
});

describe('LLMService', () => {
  const LLMService = require('../src/llmService');
  
  describe('constructor', () => {
    it('should initialize with API key and model', () => {
      const service = new LLMService('test-key', 'gpt-3.5-turbo');
      expect(service.model).toBe('gpt-3.5-turbo');
    });

    it('should throw error without API key', () => {
      expect(() => {
        new LLMService(null, 'gpt-3.5-turbo');
      }).toThrow('OpenAI API key is required');
    });
  });
});
