import os
import logging
import time
from functools import wraps
from typing import Dict, Any, Optional

from flask import Flask, render_template, request, jsonify
from flask_caching import Cache
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import system_prompt
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Configuration
app.config.update(
    PINECONE_API_KEY=config.PINECONE_API_KEY,
    OPENAI_API_KEY=config.OPENAI_API_KEY,
    CACHE_TYPE="simple",
    CACHE_DEFAULT_TIMEOUT=300
)

# Set environment variables
os.environ["PINECONE_API_KEY"] = config.PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY

# Initialize cache
cache = Cache(app)

# Initialize rate limiter with more reasonable limits
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["1000 per day", "100 per hour"]  # Increased limits
)

# Global variables for AI components
embeddings = None
docsearch = None
rag_chain = None

def initialize_ai_components():
    """Initialize AI components with error handling and retry logic."""
    global embeddings, docsearch, rag_chain
    
    try:
        logger.info("Initializing AI components...")
        
        # Initialize embeddings
        embeddings = download_hugging_face_embeddings()
        logger.info("Embeddings initialized successfully")
        
        # Initialize Pinecone vector store
        index_name = "medicalbot"
        docsearch = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings
        )
        logger.info("Pinecone vector store initialized successfully")
        
        # Initialize retriever
        retriever = docsearch.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 3}
        )
        
        # Initialize LLM
        llm = OpenAI(
            temperature=0.4, 
            max_tokens=500,
            api_key=config.OPENAI_API_KEY
        )
        
        # Initialize prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        # Create chains
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        logger.info("All AI components initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize AI components: {str(e)}")
        return False

def performance_monitor(f):
    """Decorator to monitor function performance."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        try:
            result = f(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"Function {f.__name__} executed in {execution_time:.2f} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Function {f.__name__} failed after {execution_time:.2f} seconds: {str(e)}")
            raise
    return decorated_function

def validate_input(text: str) -> bool:
    """Validate user input."""
    if not text or not text.strip():
        return False
    if len(text.strip()) > 1000:  # Limit input length
        return False
    return True

@app.route("/")
def index():
    """Render the main chat interface."""
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
@limiter.limit("30 per minute")  # Increased rate limit for better user experience
@performance_monitor
def chat():
    """Handle chat requests with improved error handling and validation."""
    try:
        # Validate request
        if not request.is_json and not request.form:
            return jsonify({"error": "Invalid request format"}), 400
        
        # Get message from request
        msg = request.form.get("msg") if request.form else request.json.get("msg")
        
        if not msg:
            return jsonify({"error": "Message is required"}), 400
        
        # Validate input
        if not validate_input(msg):
            return jsonify({"error": "Invalid input. Message must be between 1-1000 characters."}), 400
        
        logger.info(f"Processing chat request: {msg[:50]}...")
        
        # Check if AI components are initialized
        if not rag_chain:
            if not initialize_ai_components():
                return jsonify({"error": "AI service temporarily unavailable"}), 503
        
        # Process the request
        response = rag_chain.invoke({"input": msg})
        answer = response.get("answer", "I'm sorry, I couldn't generate a response.")
        
        logger.info(f"Generated response for: {msg[:50]}...")
        
        return jsonify({
            "answer": answer,
            "timestamp": time.time(),
            "status": "success"
        })
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            "error": "An error occurred while processing your request",
            "status": "error"
        }), 500

@app.route("/health")
def health_check():
    """Health check endpoint for monitoring."""
    try:
        # Check if AI components are available
        if rag_chain:
            return jsonify({
                "status": "healthy",
                "ai_components": "available",
                "timestamp": time.time()
            })
        else:
            return jsonify({
                "status": "degraded",
                "ai_components": "unavailable",
                "timestamp": time.time()
            }), 503
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }), 500

@app.route("/api/status")
def api_status():
    """API status endpoint."""
    return jsonify({
        "service": "Medical AI Chatbot",
        "version": "2.0.0",
        "status": "operational",
        "endpoints": {
            "chat": "/get",
            "health": "/health",
            "status": "/api/status"
        }
    })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(429)
def too_many_requests(error):
    """Handle rate limit errors."""
    return jsonify({"error": "Too many requests. Please try again later."}), 429

@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors."""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Initialize AI components on startup
    if initialize_ai_components():
        logger.info("Application started successfully")
        app.run(
            host="0.0.0.0", 
            port=8080, 
            debug=False,  # Disable debug mode in production
            threaded=True  # Enable threading for better performance
        )
    else:
        logger.error("Failed to initialize application. Exiting.")
        exit(1)
