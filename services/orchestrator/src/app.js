const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const morgan = require('morgan');
const promClient = require('prom-client');
require('dotenv').config();

const logger = require('./logger');
const EmbeddingClient = require('./embeddingClient');
const LLMService = require('./llmService');

// Prometheus metrics
const register = new promClient.Registry();

const httpRequestsTotal = new promClient.Counter({
  name: 'orchestrator_http_requests_total',
  help: 'Total number of HTTP requests',
  labelNames: ['method', 'route', 'status_code'],
  registers: [register]
});

const httpRequestDuration = new promClient.Histogram({
  name: 'orchestrator_http_request_duration_seconds',
  help: 'Duration of HTTP requests in seconds',
  labelNames: ['method', 'route'],
  registers: [register]
});

const chatRequestsTotal = new promClient.Counter({
  name: 'orchestrator_chat_requests_total',
  help: 'Total number of chat requests',
  labelNames: ['status'],
  registers: [register]
});

const chatResponseTime = new promClient.Histogram({
  name: 'orchestrator_chat_response_time_seconds',
  help: 'Chat response time in seconds',
  registers: [register]
});

// Add default metrics
promClient.collectDefaultMetrics({ register });

const app = express();
const PORT = process.env.ORCHESTRATOR_PORT || 5000;

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

// Request timing and metrics middleware
app.use((req, res, next) => {
  req.startTime = Date.now();
  
  // Prometheus metrics
  const end = httpRequestDuration.startTimer({
    method: req.method,
    route: req.route?.path || req.path
  });

  res.on('finish', () => {
    end();
    httpRequestsTotal.inc({
      method: req.method,
      route: req.route?.path || req.path,
      status_code: res.statusCode
    });
  });

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

// Metrics endpoint
app.get('/metrics', async (req, res) => {
  try {
    res.set('Content-Type', register.contentType);
    res.end(await register.metrics());
  } catch (error) {
    logger.error('Error generating metrics:', error);
    res.status(500).json({ error: 'Failed to generate metrics' });
  }
});

// Main chat endpoint
app.post('/chat', async (req, res) => {
  const startTime = Date.now();
  const chatTimer = chatResponseTime.startTimer();
  
  try {
    const { user_id, query, k = 5, similarity_metric = 'cosine' } = req.body;
    
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

    // Validate similarity metric
    if (!['cosine', 'dot_product'].includes(similarity_metric)) {
      return res.status(400).json({
        error: 'similarity_metric must be "cosine" or "dot_product"',
        code: 'INVALID_SIMILARITY_METRIC'
      });
    }

    logger.info(`Processing chat request for user ${user_id} with ${similarity_metric} similarity: ${query}`);

    // Step 1: Get similar documents from vector store
    const similarDocs = await embeddingClient.searchSimilar(query, k, similarity_metric);
    
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
        metadata: doc.metadata,
        similarity_metric: doc.similarity_metric || similarity_metric
      })),
      similarity_metric,
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
    chatTimer();
    chatRequestsTotal.inc({ status: 'success' });
    res.json(response);

  } catch (error) {
    const processingTime = Date.now() - startTime;
    logger.error(`Chat request failed after ${processingTime}ms:`, error);
    
    chatTimer();
    chatRequestsTotal.inc({ status: 'error' });
    
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
