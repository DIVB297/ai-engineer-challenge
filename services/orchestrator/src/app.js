const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const morgan = require('morgan');
require('dotenv').config();

const logger = require('./logger');
const EmbeddingClient = require('./embeddingClient');
const LLMService = require('./llmService');

const app = express();
const PORT = process.env.ORCHESTRATOR_PORT || 3000;

// Initialize services
const embeddingClient = new EmbeddingClient(
  process.env.EMBEDDING_SERVICE_URL || 'http://localhost:8000'
);

const llmService = new LLMService(
  process.env.OPENAI_API_KEY,
  process.env.LLM_MODEL || 'gpt-3.5-turbo'
);

// Middleware
app.use(helmet());
app.use(cors());
app.use(morgan('combined', { stream: { write: message => logger.info(message.trim()) } }));
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

// Request timing middleware
app.use((req, res, next) => {
  req.startTime = Date.now();
  next();
});

// Health check endpoint
app.get('/health', async (req, res) => {
  try {
    const embeddingHealth = await embeddingClient.healthCheck();
    
    res.json({
      status: 'healthy',
      service: 'rag-orchestrator',
      version: '1.0.0',
      timestamp: new Date().toISOString(),
      dependencies: {
        embedding_service: embeddingHealth
      }
    });
  } catch (error) {
    logger.error('Health check failed:', error);
    res.status(503).json({
      status: 'unhealthy',
      service: 'rag-orchestrator',
      error: error.message
    });
  }
});

// Main chat endpoint
app.post('/chat', async (req, res) => {
  const startTime = Date.now();
  
  try {
    const { user_id, query, k = 5 } = req.body;
    
    // Validation
    if (!query) {
      return res.status(400).json({
        error: 'Query is required',
        code: 'MISSING_QUERY'
      });
    }

    if (!user_id) {
      return res.status(400).json({
        error: 'User ID is required',
        code: 'MISSING_USER_ID'
      });
    }

    logger.info(`Processing chat request for user ${user_id}: ${query}`);

    // Step 1: Get similar documents from vector store
    const similarDocs = await embeddingClient.searchSimilar(query, k);
    
    logger.info(`Found ${similarDocs.length} similar documents`);

    // Step 2: Generate response using LLM
    const llmResponse = await llmService.generateResponse(query, similarDocs);

    // Step 3: Prepare response
    const processingTime = Date.now() - startTime;
    
    const response = {
      answer: llmResponse.answer,
      source_docs: similarDocs.map(doc => ({
        id: doc.id,
        text: doc.text.substring(0, 200) + (doc.text.length > 200 ? '...' : ''),
        score: doc.score,
        metadata: doc.metadata
      })),
      timing_ms: processingTime,
      user_id,
      query,
      model_info: {
        model: llmResponse.model,
        usage: llmResponse.usage
      },
      timestamp: new Date().toISOString()
    };

    logger.info(`Chat response generated in ${processingTime}ms for user ${user_id}`);
    res.json(response);

  } catch (error) {
    const processingTime = Date.now() - startTime;
    logger.error(`Chat request failed after ${processingTime}ms:`, error);
    
    res.status(500).json({
      error: 'Internal server error',
      message: error.message,
      timing_ms: processingTime,
      timestamp: new Date().toISOString()
    });
  }
});

// Error handling middleware
app.use((error, req, res, next) => {
  logger.error('Unhandled error:', error);
  res.status(500).json({
    error: 'Internal server error',
    message: process.env.NODE_ENV === 'development' ? error.message : 'Something went wrong'
  });
});

// 404 handler
app.use((req, res) => {
  res.status(404).json({
    error: 'Not found',
    message: `Route ${req.method} ${req.path} not found`
  });
});

// Start server
app.listen(PORT, () => {
  logger.info(`RAG Orchestrator running on port ${PORT}`);
  logger.info(`Environment: ${process.env.NODE_ENV || 'development'}`);
  logger.info(`Embedding service URL: ${process.env.EMBEDDING_SERVICE_URL || 'http://localhost:8000'}`);
});

module.exports = app;
