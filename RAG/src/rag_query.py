# src/rag_query.py
"""
RAG query module for retrieving information and generating answers.
"""

import logging
from typing import List, Dict, Any, Optional, Union

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGQueryEngine:
    """
    Engine for performing RAG queries using embeddings and ChromaDB
    """
    
    def __init__(self, 
                embedding_manager, 
                chroma_manager, 
                ollama_manager,
                document_processor=None,
                default_results: int = 3):
        """
        Initialize the RAG query engine
        
        Args:
            embedding_manager: EmbeddingManager instance
            chroma_manager: ChromaManager instance
            ollama_manager: OllamaManager instance
            document_processor: DocumentProcessor instance
            default_results: Default number of results to retrieve
        """
        self.embedding_manager = embedding_manager
        self.chroma_manager = chroma_manager
        self.ollama_manager = ollama_manager
        self.document_processor = document_processor
        self.default_results = default_results
        
        logger.info(f"RAGQueryEngine initialized with default_results={default_results}")
    
    def query(self, 
             query_text: str, 
             n_results: Optional[int] = None,
             where_filter: Optional[Dict[str, Any]] = None,
             verbose: bool = True) -> Optional[Dict[str, Any]]:
        """
        Perform a RAG query
        
        Args:
            query_text: Query text
            n_results: Number of results to retrieve (defaults to self.default_results)
            where_filter: Optional filter criteria for ChromaDB
            verbose: Whether to print detailed output
            
        Returns:
            Query results with generated answer or None if failed
        """
        n_results = n_results or self.default_results
        
        if verbose:
            logger.info(f"Query: {query_text}")
        
        # Generate embedding for query
        query_embedding = self.embedding_manager.generate_text_embedding(query_text)
        
        if not query_embedding:
            logger.error("Failed to generate query embedding")
            return None
        
        # Query vector store
        results = self.chroma_manager.query_with_embedding(
            query_embedding=query_embedding,
            n_results=n_results,
            where=where_filter
        )
        
        # Extract relevant information
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        
        if verbose:
            logger.info(f"Found {len(documents)} relevant chunks")
        
        # Build context from results
        context = ""
        contexts_info = []
        
        for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
            relevance = 1.0 - dist  # Convert distance to similarity score
            source = meta.get("source", "unknown")
            page = meta.get("page", "unknown")
            content_type = meta.get("content_type", "unknown")
            
            if verbose:
                logger.info(f"Result {i+1}: Relevance: {relevance:.4f}, "
                           f"Source: {source}, Page: {page}, Type: {content_type}")
                logger.info(f"Content preview: {doc[:100]}...")
            
            context += f"\nContext {i+1} (Relevance: {relevance:.2f}, Source: {source}, Page: {page}, Type: {content_type}):\n"
            context += f"{doc}\n"
            
            contexts_info.append({
                "text": doc,
                "metadata": meta,
                "relevance": relevance
            })
        
        # Generate answer
        prompt = f"""Based on the following context information, please answer the question.
If the context doesn't contain relevant information, please say "I don't have enough information to answer this question."

{context}

Question: {query_text}

Answer:"""
        
        answer = self.ollama_manager.generate_text(prompt, max_tokens=500)
        
        if 'error' in answer:
            logger.error(f"Error generating answer: {answer.get('error')}")
            return None
        
        if verbose:
            logger.info(f"Answer: {answer.get('response', '')}")
        
        return {
            "query": query_text,
            "answer": answer.get("response", ""),
            "contexts": contexts_info,
            "raw_response": answer
        }
    
    def filter_by_document(self, document_name: str) -> Dict[str, Any]:
        """
        Create a filter for querying a specific document
        
        Args:
            document_name: Name of document to filter by
            
        Returns:
            Filter dictionary for ChromaDB
        """
        return {"source": document_name}
    
    def filter_by_content_type(self, content_type: str) -> Dict[str, Any]:
        """
        Create a filter for querying a specific content type
        
        Args:
            content_type: Content type to filter by ('text' or 'image_description')
            
        Returns:
            Filter dictionary for ChromaDB
        """
        return {"content_type": content_type}
    
    def filter_by_page(self, page_num: int) -> Dict[str, Any]:
        """
        Create a filter for querying a specific page
        
        Args:
            page_num: Page number to filter by
            
        Returns:
            Filter dictionary for ChromaDB
        """
        return {"page": page_num}