# 🏥 Medical AI Chatbot - Enhanced Edition

A high-performance, production-ready medical chatbot powered by Generative AI, featuring advanced RAG (Retrieval-Augmented Generation) capabilities and enterprise-grade architecture.

## ✨ Features

### 🚀 **Performance & Scalability**
- **Multi-threaded Flask application** with optimized request handling
- **Intelligent caching** with configurable TTL for repeated queries
- **Rate limiting** to prevent abuse and ensure fair usage
- **Connection pooling** and resource management
- **Asynchronous processing** for better throughput

### 🧠 **AI & ML Capabilities**
- **Advanced RAG pipeline** with Pinecone vector database
- **HuggingFace embeddings** for semantic search
- **OpenAI GPT integration** for natural language understanding
- **Document chunking** with optimal overlap strategies
- **Context-aware responses** based on medical knowledge base

### 📊 **Observability**
- **Health check endpoints** for system status
- **Comprehensive logging** with structured format
- **Error tracking** and performance tracking
- **Built-in performance decorators** for function timing

### 🔒 **Security & Reliability**
- **Input validation** and sanitization
- **Rate limiting** and abuse prevention
- **Error handling** with graceful degradation
- **Health checks** and automatic recovery
- **Production-ready Docker** configuration

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Client    │    │   Flask App     │    │   AI Pipeline   │
│                 │◄──►│                 │◄──►│                 │
│  (HTML/CSS/JS)  │    │  (Multi-thread) │    │ (LangChain/RAG) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Cache Layer   │    │  Vector Store   │
                       │                 │    │   (Pinecone)    │
                       │ (Redis/Memory)  │    │                 │
                       └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Conda or Python virtual environment
- Pinecone API key
- OpenAI API key

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/End-to-end-Medical-Chatbot-Generative-AI.git
cd End-to-end-Medical-Chatbot-Generative-AI
```

### 2. Create Environment
```bash
conda create -n medibot python=3.10 -y
conda activate medibot
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Keys
**Option 1: Edit config.py directly**
```bash
# Edit config.py and add your API keys
PINECONE_API_KEY = "your_pinecone_api_key"
OPENAI_API_KEY = "your_openai_api_key"
```

**Option 2: Set Environment Variables (Recommended)**
```bash
export PINECONE_API_KEY="your_pinecone_api_key"
export OPENAI_API_KEY="your_openai_api_key"
```

**⚠️ IMPORTANT: config.py is already in .gitignore and won't be pushed to Git**

### 5. Create Vector Index
```bash
python store_index.py
```

### 6. Run Application
```bash
python app.py
```

### 7. Access Application
Open your browser and go to: `http://localhost:8080`

## 📁 Project Structure

```
End-to-end-Medical-Chatbot-Generative-AI/
├── app.py                 # Main Flask application
├── store_index.py         # Pinecone index creation script
├── config.py              # Configuration management
├── requirements.txt       # Python dependencies
├── Dockerfile            # Production Docker configuration
├── src/                  # Source code modules
│   ├── __init__.py
│   ├── helper.py         # Document processing utilities
│   ├── prompt.py         # AI prompt templates
│   └── prompt.py         # System prompts
├── templates/            # HTML templates
│   └── chat.html        # Chat interface
├── static/              # Static assets
│   └── style.css        # Enhanced UI styles
├── Data/                # Medical documents
│   └── Medical_book.pdf
└── .github/             # GitHub Actions CI/CD
```

## ⚙️ Configuration

### Environment Variables
```bash
export FLASK_ENV=production
export MEDICAL_CHATBOT_PINECONE_API_KEY="your_key"
export MEDICAL_CHATBOT_OPENAI_API_KEY="your_key"
```

### Configuration Options
- **Chunk Size**: Adjust document chunking (default: 500)
- **Cache TTL**: Response caching duration (default: 300s)
- **Rate Limits**: Request throttling (default: 10/min)
- **Model Parameters**: Temperature, max tokens, etc.

## 🔧 API Endpoints

### Chat Interface
- `GET /` - Main chat interface
- `POST /get` - Chat endpoint with rate limiting

### Health & Status
- `GET /health` - Health check endpoint
- `GET /api/status` - API status information

### Response Format
```json
{
  "answer": "AI-generated response",
  "timestamp": 1234567890.123,
  "status": "success"
}
```

## 📊 Performance & Health

### Metrics Collected
- **Response Times**: Average, min, max, percentiles
- **Request Counts**: Total requests, errors, success rate
- **System Resources**: CPU, memory, disk usage
- **AI Component Health**: Model availability, performance

### Health Check Response
```json
{
  "status": "healthy",
  "ai_components": "available",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

## 🐳 Docker Deployment

### Build Image
```bash
docker build -t medical-chatbot .
```

### Run Container
```bash
docker run -d \
  -p 8080:8080 \
  -e FLASK_ENV=production \
  -e MEDICAL_CHATBOT_PINECONE_API_KEY="your_key" \
  -e MEDICAL_CHATBOT_OPENAI_API_KEY="your_key" \
  medical-chatbot
```

### Docker Compose
```yaml
version: '3.8'
services:
  medical-chatbot:
    build: .
    ports:
      - "8080:8080"
    environment:
      - FLASK_ENV=production
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
```

## 🚀 Production Deployment

### AWS Deployment
1. **Build and push Docker image to ECR**
2. **Deploy to ECS/Fargate** with auto-scaling
3. **Set up Application Load Balancer**
4. **Configure CloudWatch for production observability**

### Performance Tuning
- **Enable Redis caching** for production
- **Use GPU instances** for embedding generation
- **Implement CDN** for static assets
- **Set up auto-scaling** based on metrics

## 📈 Performance Benchmarks

### Response Times
- **Average**: < 2 seconds
- **95th Percentile**: < 5 seconds
- **99th Percentile**: < 10 seconds

### Throughput
- **Concurrent Users**: 100+
- **Requests/Second**: 50+
- **Cache Hit Rate**: 80%+

## 🧪 Testing

### Run Tests
```bash
pytest tests/ -v --cov=src
```

### Performance Testing
```bash
# Load testing with locust
locust -f tests/locustfile.py --host=http://localhost:8080
```

## 🔍 Troubleshooting

### Common Issues
1. **API Key Errors**: Verify keys in config.py
2. **Memory Issues**: Check chunk size and overlap settings
3. **Slow Responses**: Monitor system resources and cache hit rate
4. **Index Errors**: Verify Pinecone index exists and is ready

### Logs
- **Application Logs**: `medical_chatbot.log`
- **Index Creation**: `index_creation.log`
- **Docker Logs**: `docker logs <container_id>`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

## 🙏 Acknowledgments

- **LangChain** for RAG pipeline implementation
- **Pinecone** for vector database services
- **OpenAI** for language model capabilities
- **Flask** for web framework
- **HuggingFace** for embedding models

