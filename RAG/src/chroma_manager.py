# src/chroma_manager.py
"""
ChromaDB manager module for storing and retrieving vector embeddings.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

# Add project root to path for config import
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from config import *

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChromaManager:
    """
    Manager for handling ChromaDB operations for vector storage and retrieval
    """
    
    def __init__(self, 
                 persist_directory: Optional[Path] = None,
                 embedding_model: str = "nomic-embed-text",
                 collection_name: str = "documents"):
        """
        Initialize the ChromaDB manager
        
        Args:
            persist_directory: Directory to persist ChromaDB (defaults to VECTOR_DB_PATH)
            embedding_model: Ollama model to use for embeddings
            collection_name: Name of the collection to use
        """
        self.persist_directory = persist_directory or VECTOR_DB_PATH
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        
        # Create directory if it doesn't exist
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(allow_reset=True)
        )
        
        # Initialize Ollama embedding function
        self.embedding_function = OllamaEmbeddingFunction(
            model_name=self.embedding_model,
            url="http://localhost:11434/api/embeddings"
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"Using existing collection: {self.collection_name}")
        except:
            logger.info(f"Creating new collection: {self.collection_name}")
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
        
        logger.info(f"ChromaManager initialized with collection: {self.collection_name}")
    
    def add_text(self, 
                text: str, 
                metadata: Dict[str, Any], 
                id: str) -> bool:
        """
        Add text content to the collection
        
        Args:
            text: Text content to add
            metadata: Additional metadata for the document
            id: Unique identifier for the document
            
        Returns:
            Success status
        """
        try:
            self.collection.add(
                documents=[text],
                metadatas=[metadata],
                ids=[id]
            )
            logger.info(f"Added text with ID: {id}")
            return True
        except Exception as e:
            logger.error(f"Failed to add text with ID {id}: {e}")
            return False
    
    def add_texts(self, 
                 texts: List[str], 
                 metadatas: List[Dict[str, Any]], 
                 ids: List[str]) -> bool:
        """
        Add multiple text contents to the collection
        
        Args:
            texts: List of text contents to add
            metadatas: List of metadata dictionaries
            ids: List of unique identifiers
            
        Returns:
            Success status
        """
        if not (len(texts) == len(metadatas) == len(ids)):
            logger.error(f"Lengths don't match: texts={len(texts)}, metadatas={len(metadatas)}, ids={len(ids)}")
            return False
        
        try:
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Added {len(texts)} texts to collection")
            return True
        except Exception as e:
            logger.error(f"Failed to add texts: {e}")
            return False
    
    def add_with_embedding(self,
                         text: str,
                         embedding: List[float],
                         metadata: Dict[str, Any],
                         id: str) -> bool:
        """
        Add text content with pre-computed embedding
        
        Args:
            text: Text content to add
            embedding: Pre-computed embedding vector
            metadata: Additional metadata for the document
            id: Unique identifier for the document
            
        Returns:
            Success status
        """
        try:
            self.collection.add(
                documents=[text],
                embeddings=[embedding],
                metadatas=[metadata],
                ids=[id]
            )
            logger.info(f"Added text with custom embedding, ID: {id}")
            return True
        except Exception as e:
            logger.error(f"Failed to add text with custom embedding, ID {id}: {e}")
            return False
    
    def add_with_embeddings(self,
                          texts: List[str],
                          embeddings: List[List[float]],
                          metadatas: List[Dict[str, Any]],
                          ids: List[str]) -> bool:
        """
        Add multiple text contents with pre-computed embeddings
        
        Args:
            texts: List of text contents to add
            embeddings: List of pre-computed embedding vectors
            metadatas: List of metadata dictionaries
            ids: List of unique identifiers
            
        Returns:
            Success status
        """
        if not (len(texts) == len(embeddings) == len(metadatas) == len(ids)):
            logger.error(f"Lengths don't match: texts={len(texts)}, embeddings={len(embeddings)}, "
                       f"metadatas={len(metadatas)}, ids={len(ids)}")
            return False
        
        try:
            self.collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Added {len(texts)} texts with custom embeddings")
            return True
        except Exception as e:
            logger.error(f"Failed to add texts with custom embeddings: {e}")
            return False
    
    def query(self,
             query_text: str,
             n_results: int = 3,
             where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Query the collection using text
        
        Args:
            query_text: Text to search for
            n_results: Number of results to return
            where: Optional filter criteria
            
        Returns:
            Query results
        """
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=where
            )
            logger.info(f"Query returned {len(results.get('ids', [[]])[0])} results")
            return results
        except Exception as e:
            logger.error(f"Failed to query: {e}")
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
    
    def query_with_embedding(self,
                           query_embedding: List[float],
                           n_results: int = 3,
                           where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Query the collection using a pre-computed embedding
        
        Args:
            query_embedding: Pre-computed embedding vector
            n_results: Number of results to return
            where: Optional filter criteria
            
        Returns:
            Query results
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where
            )
            logger.info(f"Embedding query returned {len(results.get('ids', [[]])[0])} results")
            return results
        except Exception as e:
            logger.error(f"Failed to query with embedding: {e}")
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection
        
        Returns:
            Collection statistics
        """
        try:
            count = self.collection.count()
            
            # Get sample item to determine embedding dimension
            sample = self.collection.get(limit=1)
            embedding_dim = "unknown"
            if sample and sample.get("embeddings") and len(sample["embeddings"]) > 0:
                embedding_dim = len(sample["embeddings"][0])
            
            stats = {
                "name": self.collection_name,
                "count": count,
                "embedding_model": self.embedding_model,
                "embedding_dimension": embedding_dim,
                "persist_directory": str(self.persist_directory)
            }
            
            logger.info(f"Collection stats: {count} items")
            return stats
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"error": str(e)}