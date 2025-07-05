# src/image_query.py
"""
Helper module for querying images and related text in RAG system.
"""

import os
import tempfile
import logging
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display, Markdown, HTML

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageQueryHelper:
    """
    Helper for image-related queries in RAG system
    """
    
    def __init__(self, 
                rag_engine, 
                chroma_manager, 
                embedding_manager):
        """
        Initialize the image query helper
        
        Args:
            rag_engine: RAGQueryEngine instance
            chroma_manager: ChromaManager instance
            embedding_manager: EmbeddingManager instance
        """
        self.rag_engine = rag_engine
        self.chroma_manager = chroma_manager
        self.embedding_manager = embedding_manager
    
    def find_images_by_query(self, query_text: str, n_results: int = 3) -> Dict[str, Any]:
        """
        Find images related to a text query
        
        Args:
            query_text: Text query to find related images
            n_results: Number of results to retrieve
            
        Returns:
            Query results with focus on images
        """
        # Create filter for image content only
        image_filter = {"content_type": "image_description"}
        
        # Generate embedding for query
        query_embedding = self.embedding_manager.generate_text_embedding(query_text)
        
        if not query_embedding:
            logger.error("Failed to generate query embedding")
            return None
        
        # Query vector store for images only
        results = self.chroma_manager.query_with_embedding(
            query_embedding=query_embedding,
            n_results=n_results,
            where=image_filter
        )
        
        # Process results
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        
        # Format results
        formatted_results = {
            "query": query_text,
            "results": [
                {
                    "description": doc,
                    "metadata": meta,
                    "relevance": 1.0 - dist,
                    "source_file": meta.get("source", ""),
                    "page": meta.get("page", "")
                }
                for doc, meta, dist in zip(documents, metadatas, distances)
            ]
        }
        
        return formatted_results
    
    def display_image_results(self, 
                             results: Dict[str, Any], 
                             show_images: bool = True,
                             show_descriptions: bool = True) -> None:
        """
        Display image query results
        
        Args:
            results: Results from find_images_by_query
            show_images: Whether to display image previews
            show_descriptions: Whether to display image descriptions
        """
        if not results or not results.get("results"):
            display(Markdown("### No image results found"))
            return
        
        query = results.get("query", "")
        display(Markdown(f"## Image Results for Query: '{query}'"))
        
        for i, result in enumerate(results.get("results", [])):
            description = result.get("description", "")
            source_file = result.get("source_file", "")
            page = result.get("page", "")
            relevance = result.get("relevance", 0.0)
            
            display(Markdown(f"### Result {i+1}"))
            display(Markdown(f"**Source**: {source_file}, **Page**: {page}, **Relevance**: {relevance:.4f}"))
            
            if show_descriptions:
                # Display preview of description
                preview_length = min(300, len(description))
                display(Markdown(f"**Description Preview**:\n{description[:preview_length]}..."))
            
            if show_images:
                display(Markdown("**Image Preview**:"))
                self.display_image_from_processed_doc(source_file, page)

        ## new code

    def display_image_from_processed_doc(self,
                                    source_file: str, 
                                    page: int) -> None:
        """
        Display an image from a processed document (handles flattened PDFs)
    
        Args:
        source_file: Source file name (may be virtual mixed name)
        page: Page number
        """
        document_processor = self.rag_engine.document_processor
    
        try:
            # Map virtual filenames to actual PDF files
            actual_filename = self._map_to_actual_filename(source_file)
        
        # Find the document in data directory
            data_dir = document_processor.config.get('data_dir')
            file_path = os.path.join(data_dir, actual_filename)
        
            if not os.path.exists(file_path):
                display(Markdown(f"*Image not found: {file_path}*"))
                return
        
            # Convert PDF page to image (for flattened PDFs)
            page_image = self._convert_pdf_page_to_image(file_path, page, document_processor)
        
            if page_image:
                # Display the page image
                plt.figure(figsize=(12, 10))
                plt.imshow(page_image)
                plt.axis('off')
                plt.title(f"Page {page} from {actual_filename}")
                plt.show()
            else:
                display(Markdown(f"*Failed to convert page {page} of {actual_filename} to image*"))
    
        except Exception as e:
            display(Markdown(f"*Error displaying image: {str(e)}*"))

    def _map_to_actual_filename(self, source_file: str) -> str:
        """
        Map virtual mixed filenames to actual PDF files
    
        Args:
            source_file: Virtual or actual filename
        
        Returns:
            Actual PDF filename
        """
        # Map virtual mixed names to actual image PDF
        filename_mappings = {
            "mixed_singapore_explorer_guide_text1": "singapore_explorer_guide_image1.pdf",
            "mixed_singapore_explorer_guide_text": "singapore_explorer_guide_image1.pdf"
        }
    
        return filename_mappings.get(source_file, source_file)

    def _convert_pdf_page_to_image(self, file_path: str, page_num: int, document_processor) -> Optional[Image.Image]:
        """
        Convert a specific PDF page to PIL Image (for flattened PDFs)
    
        Args:
        file_path: Path to PDF file
        page_num: Page number (1-based)
        document_processor: DocumentProcessor instance
        
        Returns:
        PIL Image object or None if failed
        """
        try:
            import fitz
            import io
            from PIL import Image
        
            # Open PDF document
            doc = fitz.open(file_path)
        
            # Check if page exists
            if page_num < 1 or page_num > len(doc):
                doc.close()
                return None
        
            # Load the specific page (convert to 0-based index)
            page = doc.load_page(page_num - 1)
        
            # Convert page to high-resolution image
            mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
            pix = page.get_pixmap(matrix=mat)
        
            # Convert to PIL Image
            img_data = pix.tobytes("png")
            pil_image = Image.open(io.BytesIO(img_data))
        
            # Resize if too large
            max_size = document_processor.config.get('max_image_size', 1024)
            if max(pil_image.size) > max_size:
                pil_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
            # Clean up
            doc.close()
            pix = None
        
            return pil_image
        
        except Exception as e:
            logger.error(f"Error converting PDF page to image: {str(e)}")
            return None
    
    ## old code below

    
    def ask_about_image(self, 
                       source_file: str, 
                       page: int, 
                       query_text: str) -> Dict[str, Any]:
        """
        Ask a question about a specific image
        
        Args:
            source_file: Source file name
            page: Page number
            query_text: Question about the image
            
        Returns:
            Query result with answer about the image
        """
        # Load the image
        document_processor = self.rag_engine.document_processor
        data_dir = document_processor.config.get('data_dir')
        file_path = os.path.join(data_dir, source_file)
        
        try:
            # Process document to get images
            processed_doc = document_processor.process_document(file_path)
            
            # Find image content for the specified page
            image_contents = [
                content for content in processed_doc.extracted_contents
                if content.content_type.value == 'image' and content.source_page == page
            ]
            
            if not image_contents:
                return {"error": f"No image found on page {page} of {source_file}"}
            
            # Get the image
            image = image_contents[0].content_data
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                temp_path = temp_file.name
                image.save(temp_path)
            
            try:
                # Use direct image processing with the vision model
                ollama_manager = self.rag_engine.ollama_manager
                
                # Prepare prompt with specific question
                prompt = f"Look at this image and answer the following question: {query_text}"
                
                # Process image with question
                result = ollama_manager.process_image_text(str(temp_path), prompt)
                
                if 'error' in result:
                    return {"error": f"Failed to process image: {result['error']}"}
                
                return {
                    "query": query_text,
                    "image_source": {
                        "file": source_file,
                        "page": page
                    },
                    "answer": result.get("response", ""),
                    "raw_response": result
                }
                
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
        except Exception as e:
            return {"error": f"Error processing image: {str(e)}"}
    
    def display_image_query_result(self, 
                                 result: Dict[str, Any],
                                 show_image: bool = True) -> None:
        """
        Display the result of an image query
        
        Args:
            result: Result from ask_about_image
            show_image: Whether to display the image
        """
        if 'error' in result:
            display(Markdown(f"### ❌ Error: {result['error']}"))
            return
        
        query = result.get("query", "")
        answer = result.get("answer", "")
        image_source = result.get("image_source", {})
        
        source_file = image_source.get("file", "")
        page = image_source.get("page", "")
        
        display(Markdown(f"## Image Query Result"))
        display(Markdown(f"**Question**: {query}"))
        
        if show_image:
            display(Markdown("**Image**:"))
            self.display_image_from_processed_doc(source_file, page)
        
        display(Markdown(f"**Answer**:\n{answer}"))

    def display_image_result_dict(self, image_result_dict: Dict[str, Any]) -> None:
        """
        Display a summary of multiple image query results for evaluation
        
        Args:
            image_result_dict: Dictionary of image query results
        """
        if not image_result_dict:
            display(Markdown("### No image results to display"))
            return    
        
        # Display summary table
        display(Markdown("# Image Query Results Summary"))
        display(Markdown("---"))
        
        # Create summary table
        table_rows = []
        table_rows.append("| Query ID | Type | Query/Question | Source | Page | Key Info |")
        table_rows.append("|----------|------|----------------|--------|------|----------|")
        
        for query_id, result in image_result_dict.items():
            if 'error' in result:
                table_rows.append(f"| {query_id} | Error | - | - | - | {result['error']} |")
                continue
            
            # Determine result type and extract info
            if 'results' in result:
                # Search result type (from find_images_by_query)
                query_type = "🔍 Search"
                query_text = result.get('query', 'Unknown')
                results = result.get('results', [])
                
                if results:
                    first_result = results[0]
                    source_file = first_result.get('source_file', '-')
                    page = first_result.get('page', '-')
                    relevance = first_result.get('relevance', 0.0)
                    key_info = f"{len(results)} results (best: {relevance:.3f})"
                else:
                    source_file = page = "-"
                    key_info = "No results"
                    
            elif 'answer' in result:
                # Direct image query type (from ask_about_image)
                query_type = "❓ Direct"
                query_text = result.get('query', 'Unknown')
                image_source = result.get('image_source', {})
                source_file = image_source.get('file', '-')
                page = image_source.get('page', '-')
                answer = result.get('answer', '')
                key_info = f"Answer: {answer[:50]}..." if len(answer) > 50 else f"Answer: {answer}"
            else:
                # Unknown type
                query_type = "❔ Unknown"
                query_text = "Unknown format"
                source_file = page = "-"
                key_info = "Unknown result format"
            
            # Truncate query text if too long
            if len(query_text) > 40:
                query_text = query_text[:37] + "..."
            
            table_rows.append(f"| {query_id} | {query_type} | {query_text} | {source_file} | {page} | {key_info} |")
        
        # Display the table
        display(Markdown("\n".join(table_rows)))
        
        # Display detailed results
        display(Markdown("\n## Detailed Results"))
        display(Markdown("---"))
        
        for query_id, result in image_result_dict.items():
            display(Markdown(f"### 📋 {query_id}"))
            
            if 'error' in result:
                display(Markdown(f"**❌ Error:** {result['error']}"))
                continue
            
            if 'results' in result:
                # Handle search results
                query = result.get('query', 'Unknown query')
                results = result.get('results', [])
                
                display(Markdown(f"**🔍 Search Query:** {query}"))
                display(Markdown(f"**📊 Results Found:** {len(results)}"))
                
                if results:
                    display(Markdown("**🎯 Top Results:**"))
                    for i, res in enumerate(results[:2]):  # Show top 2 results
                        source_file = res.get('source_file', '-')
                        page = res.get('page', '-')
                        relevance = res.get('relevance', 0.0)
                        description = res.get('description', '')
                        
                        # Show brief description preview
                        desc_preview = description[:150] + "..." if len(description) > 150 else description
                        
                        display(Markdown(f"**Result {i+1}:** {source_file}, Page {page} (Relevance: {relevance:.4f})"))
                        display(Markdown(f"*{desc_preview}*"))
                        
            elif 'answer' in result:
                # Handle direct image queries
                query = result.get('query', 'Unknown query')
                answer = result.get('answer', 'No answer')
                image_source = result.get('image_source', {})
                source_file = image_source.get('file', '-')
                page = image_source.get('page', '-')
                
                display(Markdown(f"**❓ Question:** {query}"))
                display(Markdown(f"**📄 Source:** {source_file}, Page {page}"))
                display(Markdown(f"**💡 Answer:**"))
                display(Markdown(f"{answer}"))
            
            display(Markdown("---"))
        
        display(Markdown(f"**📈 Summary:** {len(image_result_dict)} total queries processed"))

