# src/rag_manager.py
"""
RAG (Retrieval Augmented Generation) manager module for integrating
document processing, embedding generation, and vector storage.
"""

import os
import json
import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import time

# Add project root to path for config import
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from config import *
from src.document_processor import DocumentProcessor, ContentType, FileFormat
from src.embedding_manager import EmbeddingManager
from src.chroma_manager import ChromaManager
from src.model_manager import OllamaManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGManager:
    """
    Manager for the complete RAG pipeline
    """
    
    def __init__(self, 
                text_embedding_model: str = "nomic-embed-text",
                vision_model: str = "qwen2.5vl:3b",
                generation_model: str = "qwen2.5vl:3b",
                collection_name: str = "documents"):
        """
        Initialize the RAG manager
        
        Args:
            text_embedding_model: Model to use for text embeddings
            vision_model: Model to use for image descriptions
            generation_model: Model to use for final answer generation
            collection_name: Name of the collection in ChromaDB
        """
        self.text_embedding_model = text_embedding_model
        self.vision_model = vision_model
        self.generation_model = generation_model
        self.collection_name = collection_name
        
        # Initialize components
        self.document_processor = DocumentProcessor()
        self.embedding_manager = EmbeddingManager(
            text_embedding_model=text_embedding_model,
            vision_model=vision_model
        )
        self.chroma_manager = ChromaManager(
            embedding_model=text_embedding_model,
            collection_name=collection_name
        )
        self.ollama_manager = OllamaManager(model_name=generation_model)
        
        logger.info(f"RAGManager initialized with models: "
                  f"text={text_embedding_model}, vision={vision_model}, generation={generation_model}")
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        Process a document and add to vector store
        
        Args:
            file_path: Path to document file
            
        Returns:
            Processing statistics
        """
        start_time = time.time()
        logger.info(f"Processing document: {file_path}")
        
        # Process document
        processed_doc = self.document_processor.process_document(file_path)
        
        # Split content by type
        text_contents = [c for c in processed_doc.extracted_contents 
                       if c.content_type == ContentType.TEXT]
        image_contents = [c for c in processed_doc.extracted_contents 
                        if c.content_type == ContentType.IMAGE]
        
        logger.info(f"Document processed: {len(text_contents)} text chunks, "
                  f"{len(image_contents)} image chunks")
        
        # Process text content
        text_embeddings = []
        text_ids = []
        text_documents = []
        text_metadatas = []
        
        for content in text_contents:
            text = content.content_data
            content_id = content.content_id
            
            # Generate embedding
            embedding = self.embedding_manager.generate_text_embedding(text)
            
            if embedding:
                text_embeddings.append(embedding)
                text_ids.append(content_id)
                text_documents.append(text)
                text_metadatas.append({
                    "source": processed_doc.filename,
                    "page": content.source_page,
                    "content_type": "text",
                    "document_type": processed_doc.document_type.value
                })
        
        # Process image content
        image_embeddings = []
        image_ids = []
        image_documents = []
        image_metadatas = []
        
        for content in image_contents:
            # Save image to temp file
            image = content.content_data
            content_id = content.content_id
            
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                temp_path = temp_file.name
                image.save(temp_path)
            
            try:
                # Generate description
                description = self.embedding_manager.generate_image_description(temp_path)
                
                if description:
                    # Generate embedding for description
                    embedding = self.embedding_manager.generate_text_embedding(description)
                    
                    if embedding:
                        image_embeddings.append(embedding)
                        image_ids.append(content_id)
                        image_documents.append(description)
                        image_metadatas.append({
                            "source": processed_doc.filename,
                            "page": content.source_page,
                            "content_type": "image_description",
                            "document_type": processed_doc.document_type.value
                        })
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        
        # Add to vector store
        all_embeddings = text_embeddings + image_embeddings
        all_ids = text_ids + image_ids
        all_documents = text_documents + image_documents
        all_metadatas = text_metadatas + image_metadatas
        
        success = False
        if all_embeddings:
            success = self.chroma_manager.add_with_embeddings(
                texts=all_documents,
                embeddings=all_embeddings,
                metadatas=all_metadatas,
                ids=all_ids
            )
        
        # Collect statistics
        end_time = time.time()
        processing_time = end_time - start_time
        
        stats = {
            "filename": processed_doc.filename,
            "document_type": processed_doc.document_type.value,
            "page_count": processed_doc.document_metadata.page_count,
            "text_chunks": len(text_contents),
            "text_chunks_embedded": len(text_embeddings),
            "image_chunks": len(image_contents),
            "image_chunks_embedded": len(image_embeddings),
            "total_chunks_embedded": len(all_embeddings),
            "vector_store_success": success,
            "processing_time_seconds": processing_time
        }
        
        logger.info(f"Document processing completed in {processing_time:.1f} seconds")
        return stats
    
    def query(self, query_text: str, n_results: int = 3) -> Dict[str, Any]:
        """
        Query the RAG system
        
        Args:
            query_text: Query text
            n_results: Number of results to retrieve
            
        Returns:
            Query results with generated answer
        """
        start_time = time.time()
        
        # Generate embedding for query
        query_embedding = self.embedding_manager.generate_text_embedding(query_text)
        
        if not query_embedding:
            logger.error("Failed to generate query embedding")
            return {"error": "Failed to generate query embedding"}
        
        # Query vector store
        results = self.chroma_manager.query_with_embedding(
            query_embedding=query_embedding,
            n_results=n_results
        )
        
        # Extract relevant information
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        
        # Create context from results
        context = ""
        for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
            relevance = 1.0 - dist  # Convert distance to similarity score
            source = meta.get("source", "unknown")
            page = meta.get("page", "unknown")
            content_type = meta.get("content_type", "unknown")
            
            context += f"\nContext {i+1} (Relevance: {relevance:.2f}, Source: {source}, Page: {page}, Type: {content_type}):\n"
            context += f"{doc}\n"
        
        # Generate answer
        prompt = f"""Based on the following context information, please answer the question.
If the context doesn't contain relevant information, please say "I don't have enough information to answer this question."

{context}

Question: {query_text}

Answer:"""
        
        answer = self.ollama_manager.generate_text(prompt, max_tokens=500)
        
        # Prepare response
        end_time = time.time()
        query_time = end_time - start_time
        
        response = {
            "query": query_text,
            "answer": answer.get("response", ""),
            "contexts": [
                {
                    "text": doc,
                    "metadata": meta,
                    "relevance": 1.0 - dist
                }
                for doc, meta, dist in zip(documents, metadatas, distances)
            ],
            "query_time_seconds": query_time
        }
        
        logger.info(f"Query completed in {query_time:.1f} seconds")
        return response