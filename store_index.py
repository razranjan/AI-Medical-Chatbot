#!/usr/bin/env python3
"""
Medical Chatbot - Pinecone Index Creation Script

This script creates and populates a Pinecone index with medical document embeddings.
It includes error handling, progress tracking, and validation.
"""

import os
import sys
import logging
import time
from typing import Optional, Dict, Any
from pathlib import Path

from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from src.helper import (
    document_processor, 
    embedding_manager, 
    get_processing_stats,
    cleanup_resources
)
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('index_creation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class PineconeIndexManager:
    """Manages Pinecone index operations."""
    
    def __init__(self, api_key: str, index_name: str = "medicalbot"):
        self.api_key = api_key
        self.index_name = index_name
        self.pinecone_client = None
        self.vector_store = None
        
    def initialize_client(self) -> bool:
        """Initialize Pinecone client."""
        try:
            logger.info("Initializing Pinecone client...")
            self.pinecone_client = Pinecone(api_key=self.api_key)
            logger.info("Pinecone client initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone client: {str(e)}")
            return False
    
    def check_index_exists(self) -> bool:
        """Check if the index already exists."""
        try:
            if not self.pinecone_client:
                return False
            
            # List all indexes
            indexes = self.pinecone_client.list_indexes()
            return self.index_name in indexes
            
        except Exception as e:
            logger.error(f"Error checking index existence: {str(e)}")
            return False
    
    def create_index(self, dimension: int = 384) -> bool:
        """Create a new Pinecone index."""
        try:
            if self.check_index_exists():
                logger.info(f"Index '{self.index_name}' already exists")
                return True
            
            logger.info(f"Creating new index '{self.index_name}' with dimension {dimension}")
            
            # Create index with serverless specification
            self.pinecone_client.create_index(
                name=self.index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            
            logger.info(f"Index '{self.index_name}' created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create index: {str(e)}")
            return False
    
    def wait_for_index_ready(self, timeout: int = 300) -> bool:
        """Wait for the index to be ready for operations."""
        try:
            logger.info("Waiting for index to be ready...")
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                try:
                    # Try to describe the index
                    index_stats = self.pinecone_client.describe_index(self.index_name)
                    if index_stats.status.ready:
                        logger.info("Index is ready for operations")
                        return True
                    
                    logger.info("Index not ready yet, waiting...")
                    time.sleep(10)
                    
                except Exception as e:
                    logger.warning(f"Error checking index status: {str(e)}")
                    time.sleep(10)
            
            logger.error(f"Index not ready after {timeout} seconds")
            return False
            
        except Exception as e:
            logger.error(f"Error waiting for index: {str(e)}")
            return False
    
    def populate_index(self, documents: list, embeddings) -> bool:
        """Populate the index with document embeddings."""
        try:
            if not documents:
                logger.warning("No documents provided for index population")
                return False
            
            logger.info(f"Populating index with {len(documents)} documents...")
            
            # Create vector store and populate index
            self.vector_store = PineconeVectorStore.from_documents(
                documents=documents,
                index_name=self.index_name,
                embedding=embeddings
            )
            
            logger.info("Index populated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to populate index: {str(e)}")
            return False
    
    def get_index_stats(self) -> Optional[Dict[str, Any]]:
        """Get statistics about the index."""
        try:
            if not self.pinecone_client:
                return None
            
            stats = self.pinecone_client.describe_index(self.index_name)
            return {
                "name": stats.name,
                "dimension": stats.dimension,
                "metric": stats.metric,
                "status": stats.status.state,
                "total_vector_count": stats.status.total_vector_count
            }
            
        except Exception as e:
            logger.error(f"Error getting index stats: {str(e)}")
            return None

def validate_environment() -> bool:
    """Validate that all required environment variables are set."""
    required_vars = ['PINECONE_API_KEY']
    
    for var in required_vars:
        if not hasattr(config, var) or not getattr(config, var):
            logger.error(f"Required environment variable {var} is not set")
            return False
    
    logger.info("Environment validation passed")
    return True

def process_documents(data_path: str) -> tuple:
    """Process documents and return documents and chunks."""
    try:
        logger.info(f"Processing documents from: {data_path}")
        
        # Load PDF files
        documents = document_processor.load_pdf_files(data_path)
        if not documents:
            logger.error("No documents loaded")
            return None, None
        
        # Split into chunks
        chunks = document_processor.split_text(documents)
        if not chunks:
            logger.error("No chunks created")
            return None, None
        
        # Validate chunks
        if not document_processor.validate_chunks(chunks):
            logger.warning("Some chunks failed validation")
        
        # Get processing statistics
        stats = get_processing_stats(documents, chunks)
        logger.info(f"Document processing completed: {stats}")
        
        return documents, chunks
        
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}")
        return None, None

def main():
    """Main function to create and populate the Pinecone index."""
    start_time = time.time()
    
    try:
        logger.info("Starting Pinecone index creation process...")
        
        # Validate environment
        if not validate_environment():
            logger.error("Environment validation failed")
            sys.exit(1)
        
        # Process documents
        data_path = 'Data/'
        documents, chunks = process_documents(data_path)
        
        if not documents or not chunks:
            logger.error("Document processing failed")
            sys.exit(1)
        
        # Initialize Pinecone manager
        index_manager = PineconeIndexManager(
            api_key=config.PINECONE_API_KEY,
            index_name="medicalbot"
        )
        
        # Initialize Pinecone client
        if not index_manager.initialize_client():
            logger.error("Failed to initialize Pinecone client")
            sys.exit(1)
        
        # Get embedding dimension
        embeddings = embedding_manager.get_embeddings()
        dimension = embedding_manager.get_embedding_dimension()
        logger.info(f"Using embedding dimension: {dimension}")
        
        # Create index
        if not index_manager.create_index(dimension=dimension):
            logger.error("Failed to create index")
            sys.exit(1)
        
        # Wait for index to be ready
        if not index_manager.wait_for_index_ready():
            logger.error("Index not ready within timeout")
            sys.exit(1)
        
        # Populate index
        if not index_manager.populate_index(chunks, embeddings):
            logger.error("Failed to populate index")
            sys.exit(1)
        
        # Get final statistics
        index_stats = index_manager.get_index_stats()
        if index_stats:
            logger.info(f"Index statistics: {index_stats}")
        
        # Calculate total time
        total_time = time.time() - start_time
        logger.info(f"Index creation completed successfully in {total_time:.2f} seconds")
        
        # Cleanup resources
        cleanup_resources()
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        cleanup_resources()
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        cleanup_resources()
        sys.exit(1)

if __name__ == "__main__":
    main()
