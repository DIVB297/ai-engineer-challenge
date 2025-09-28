import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [query, setQuery] = useState('');
  const [userId, setUserId] = useState('demo_user');
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState(null);
  const [error, setError] = useState(null);
  const [similarityMetric, setSimilarityMetric] = useState('cosine');
  const [k, setK] = useState(3);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError(null);
    setResponse(null);

    try {
      // Use environment variable for backend URL, fallback to localhost
      const baseURL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:5000';
      
      const result = await axios.post(`${baseURL}/chat`, {
        user_id: userId,
        query: query,
        k: k,
        similarity_metric: similarityMetric
      });

      console.log('Response data:', result.data);
      console.log('Selected similarity metric:', similarityMetric);
      setResponse(result.data);
    } catch (err) {
      setError(err.response?.data?.message || err.message || 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const formatTime = (ms) => {
    return ms < 1000 ? `${ms}ms` : `${(ms / 1000).toFixed(2)}s`;
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🤖 RAG System Demo</h1>
        <p>Ask questions and get context-aware answers from our knowledge base</p>
      </header>

      <main className="App-main">
        <form onSubmit={handleSubmit} className="query-form">
          <div className="form-group">
            <label htmlFor="userId">User ID:</label>
            <input
              id="userId"
              type="text"
              value={userId}
              onChange={(e) => setUserId(e.target.value)}
              className="form-input"
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="k">Number of sources (k):</label>
            <input
              id="k"
              type="number"
              min="1"
              max="10"
              value={k}
              onChange={(e) => setK(parseInt(e.target.value))}
              className="form-input"
            />
          </div>

          <div className="form-group">
            <label htmlFor="similarity-metric">Similarity Metric:</label>
            <select
              id="similarity-metric"
              value={similarityMetric}
              onChange={(e) => setSimilarityMetric(e.target.value)}
              className="form-input"
            >
              <option value="cosine">🎯 Cosine Similarity</option>
              <option value="dot_product">⚡ Dot Product</option>
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="query">Your Question:</label>
            <textarea
              id="query"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask anything about AI, machine learning, or technology..."
              className="form-textarea"
              rows="3"
              required
            />
          </div>

          <button 
            type="submit" 
            disabled={loading || !query.trim()}
            className="submit-button"
          >
            {loading ? '🤔 Thinking...' : '🚀 Ask Question'}
          </button>
        </form>

        {error && (
          <div className="error-container">
            <h3>❌ Error</h3>
            <p>{error}</p>
          </div>
        )}

        {response && (
          <div className="response-container">
            <div className="response-header">
              <h3>💡 Answer</h3>
              <div className="response-meta">
                <span className="timing">⏱️ {formatTime(response.timing_ms)}</span>
                <span className="model">🤖 {response.model_info?.model}</span>
                <span className="tokens">🎯 {response.model_info?.usage?.total_tokens} tokens</span>
                <span className="similarity">📊 {(response.similarity_metric || similarityMetric) === 'cosine' ? 'Cosine Similarity' : 'Dot Product'}</span>
              </div>
            </div>
            
            <div className="answer">
              <p>{response.answer}</p>
            </div>

            {response.source_docs && response.source_docs.length > 0 && (
              <div className="sources-container">
                <h4>📚 Sources ({response.source_docs.length})</h4>
                <div className="sources-list">
                  {response.source_docs.map((doc, index) => (
                    <div key={doc.id || index} className="source-item">
                      <div className="source-header">
                        <span className="source-id">{doc.id}</span>
                        <span className="source-score">
                          📊 {(doc.score * 100).toFixed(1)}% match
                        </span>
                      </div>
                      <div className="source-text">
                        {doc.text}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </main>

      <footer className="App-footer">
        <p>
          🔧 Powered by OpenAI, and MongoDB Atlas | 
          ⚡ Built with React
        </p>
      </footer>
    </div>
  );
}

export default App;
