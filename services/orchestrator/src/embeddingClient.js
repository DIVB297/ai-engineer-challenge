const axios = require('axios');
const logger = require('./logger');

class EmbeddingClient {
  constructor(baseUrl) {
    this.baseUrl = baseUrl;
    this.client = axios.create({
      baseURL: baseUrl,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json'
      }
    });
  }

  async getEmbedding(text) {
    try {
      const response = await this.client.get('/search', {
        params: { query: text, k: 1 }
      });
      
      // This is a workaround - in a real scenario, we'd have a dedicated endpoint
      // For now, we'll make a direct call to get the embedding
      const embedResponse = await this.client.post('/embed', {
        id: `query_${Date.now()}`,
        text: text
      });
      
      return embedResponse.data.embedding;
    } catch (error) {
      logger.error('Error getting embedding:', error.message);
      throw new Error(`Failed to get embedding: ${error.message}`);
    }
  }

  async searchSimilar(query, k = 5) {
    try {
      const response = await this.client.get('/search', {
        params: { query, k }
      });
      
      return response.data.results;
    } catch (error) {
      logger.error('Error searching similar documents:', error.message);
      throw new Error(`Failed to search similar documents: ${error.message}`);
    }
  }

  async healthCheck() {
    try {
      const response = await this.client.get('/health');
      return response.data;
    } catch (error) {
      logger.error('Embedding service health check failed:', error.message);
      return { status: 'unhealthy', error: error.message };
    }
  }
}

module.exports = EmbeddingClient;
