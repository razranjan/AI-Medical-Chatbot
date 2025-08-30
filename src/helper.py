import os
import logging
from typing import List, Optional
from functools import lru_cache

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# Configure logging
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Class to handle document processing operations."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 20):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_pdf_files(self, data_path: str) -> List[Document]:
        """
        Extract data from PDF files in the specified directory.
        
        Args:
            data_path (str): Path to directory containing PDF files
            
        Returns:
            List[Document]: List of loaded documents
            
        Raises:
            FileNotFoundError: If the data path doesn't exist
            Exception: If there's an error loading documents
        """
        try:
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data path '{data_path}' does not exist")
            
            logger.info(f"Loading PDF files from: {data_path}")
            
            # Check if directory contains PDF files
            pdf_files = [f for f in os.listdir(data_path) if f.lower().endswith('.pdf')]
            if not pdf_files:
                logger.warning(f"No PDF files found in {data_path}")
                return []
            
            logger.info(f"Found {len(pdf_files)} PDF files")
            
            # Load documents
            loader = DirectoryLoader(
                data_path,
                glob="*.pdf",
                loader_cls=PyPDFLoader,
                show_progress=True
            )
            
            documents = loader.load()
            logger.info(f"Successfully loaded {len(documents)} documents")
            
            return documents
            
        except Exception as e:
            logger.error(f"Error loading PDF files: {str(e)}")
            raise
    
    def split_text(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into text chunks for better processing.
        
        Args:
            documents (List[Document]): List of documents to split
            
        Returns:
            List[Document]: List of text chunks
        """
        try:
            if not documents:
                logger.warning("No documents provided for text splitting")
                return []
            
            logger.info(f"Splitting {len(documents)} documents into chunks")
            
            # Split documents
            text_chunks = self.text_splitter.split_documents(documents)
            
            logger.info(f"Successfully created {len(text_chunks)} text chunks")
            
            # Log chunk statistics
            total_chars = sum(len(chunk.page_content) for chunk in text_chunks)
            avg_chunk_size = total_chars / len(text_chunks) if text_chunks else 0
            
            logger.info(f"Average chunk size: {avg_chunk_size:.1f} characters")
            
            return text_chunks
            
        except Exception as e:
            logger.error(f"Error splitting text: {str(e)}")
            raise
    
    def validate_chunks(self, chunks: List[Document]) -> bool:
        """
        Validate that text chunks meet quality standards.
        
        Args:
            chunks (List[Document]): List of text chunks to validate
            
        Returns:
            bool: True if chunks are valid, False otherwise
        """
        if not chunks:
            return False
        
        # Check for minimum chunk size
        min_chunk_size = 50
        valid_chunks = [chunk for chunk in chunks if len(chunk.page_content.strip()) >= min_chunk_size]
        
        if len(valid_chunks) != len(chunks):
            logger.warning(f"Filtered out {len(chunks) - len(valid_chunks)} chunks below minimum size")
        
        return len(valid_chunks) > 0

class EmbeddingManager:
    """Class to manage embedding operations."""
    
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.model_name = model_name
        self._embeddings = None
    
    @lru_cache(maxsize=1)
    def get_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Get or create HuggingFace embeddings with caching.
        
        Returns:
            HuggingFaceEmbeddings: Initialized embeddings model
        """
        if self._embeddings is None:
            try:
                logger.info(f"Initializing embeddings model: {self.model_name}")
                
                self._embeddings = HuggingFaceEmbeddings(
                    model_name=self.model_name,
                    model_kwargs={'device': 'cpu'},  # Use CPU for compatibility
                    encode_kwargs={'normalize_embeddings': True}
                )
                
                logger.info("Embeddings model initialized successfully")
                
            except Exception as e:
                logger.error(f"Error initializing embeddings model: {str(e)}")
                raise
        
        return self._embeddings
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embeddings.
        
        Returns:
            int: Embedding dimension
        """
        try:
            # Test embedding to get dimension
            test_text = "test"
            embeddings = self.get_embeddings()
            test_embedding = embeddings.embed_query(test_text)
            return len(test_embedding)
        except Exception as e:
            logger.error(f"Error getting embedding dimension: {str(e)}")
            # Return default dimension for the model
            return 384

# Create global instances
document_processor = DocumentProcessor()
embedding_manager = EmbeddingManager()

# Backward compatibility functions
def load_pdf_file(data: str) -> List[Document]:
    """Backward compatibility function for loading PDF files."""
    return document_processor.load_pdf_files(data)

def text_split(extracted_data: List[Document]) -> List[Document]:
    """Backward compatibility function for splitting text."""
    return document_processor.split_text(extracted_data)

def download_hugging_face_embeddings() -> HuggingFaceEmbeddings:
    """Backward compatibility function for getting embeddings."""
    return embedding_manager.get_embeddings()

# Utility functions
def get_processing_stats(documents: List[Document], chunks: List[Document]) -> dict:
    """
    Get statistics about document processing.
    
    Args:
        documents (List[Document]): Original documents
        chunks (List[Document]): Processed chunks
        
    Returns:
        dict: Processing statistics
    """
    total_docs = len(documents)
    total_chunks = len(chunks)
    total_chars = sum(len(doc.page_content) for doc in documents)
    total_chunk_chars = sum(len(chunk.page_content) for chunk in chunks)
    
    return {
        "total_documents": total_docs,
        "total_chunks": total_chunks,
        "total_characters": total_chars,
        "total_chunk_characters": total_chunk_chars,
        "average_chunk_size": total_chunk_chars / total_chunks if total_chunks > 0 else 0,
        "compression_ratio": total_chunks / total_docs if total_docs > 0 else 0
    }

def cleanup_resources():
    """Clean up resources and clear caches."""
    try:
        # Clear LRU cache
        embedding_manager.get_embeddings.cache_clear()
        logger.info("Resources cleaned up successfully")
    except Exception as e:
        logger.error(f"Error cleaning up resources: {str(e)}")