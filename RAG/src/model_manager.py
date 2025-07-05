# src/model_manager.py
"""
Model management module for Qwen 2.5 VL with Ollama integration.
Provides helper functions for loading models, generating embeddings,
and processing multimodal content.
"""

import os
import json
import requests
import subprocess
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import torch
from PIL import Image
import io
import base64
import numpy as np

# Add project root to path for config import
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from config import *

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OllamaManager:
    """Manage Ollama service and model operations"""
    
    def __init__(self, model_name: str = "qwen2.5-vl"):
        self.model_name = model_name
        self.base_url = "http://localhost:11434/api"
        self.model_loaded = False
        logger.info(f"OllamaManager initialized with model: {model_name}")
    
    def install_ollama(self) -> bool:
        """Install Ollama if not already installed"""
        try:
            # Check if ollama is already installed
            result = subprocess.run(['ollama', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"Ollama already installed: {result.stdout.strip()}")
                return True
        except FileNotFoundError:
            pass
        
        logger.info("Installing Ollama...")
        try:
            # Install Ollama (macOS/Linux)
            subprocess.run(['curl', '-fsSL', 'https://ollama.com/install.sh'], check=True)
            logger.info("Ollama installed successfully")
            return True
        except Exception as e:
            logger.error(f"Ollama installation failed: {e}")
            return False
    
    def check_ollama_service(self) -> bool:
        """Check if Ollama service is running"""
        try:
            response = requests.get(f'{self.base_url}/tags', timeout=5)
            if response.status_code == 200:
                logger.info("Ollama service is running")
                return True
        except:
            pass
        
        logger.info("Starting Ollama service...")
        try:
            # Start Ollama service
            subprocess.Popen(['ollama', 'serve'])
            time.sleep(3)  # Wait for service to start
            
            # Check again
            response = requests.get(f'{self.base_url}/tags', timeout=5)
            if response.status_code == 200:
                logger.info("Ollama service started successfully")
                return True
        except Exception as e:
            logger.error(f"Failed to start Ollama service: {e}")
            return False
        
        return False
    
    def pull_model(self) -> bool:
        """Pull Qwen model from Ollama repository"""
        logger.info(f"Pulling model: {self.model_name}")
        
        try:
            response = requests.post(
                f'{self.base_url}/pull',
                json={"name": self.model_name},
                stream=True
            )
            
            # Process streaming response
            for line in response.iter_lines():
                if line:
                    update = json.loads(line.decode('utf-8'))
                    if 'status' in update:
                        logger.info(f"Pull status: {update['status']}")
                    if 'completed' in update and update['completed']:
                        self.model_loaded = True
                        logger.info(f"Model {self.model_name} pulled successfully")
                        return True
            
            return self.model_loaded
            
        except Exception as e:
            logger.error(f"Failed to pull model: {e}")
            return False
    
    def check_model_available(self) -> bool:
        """Check if model is available locally"""
        try:
            response = requests.get(f'{self.base_url}/tags', timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                for model in models:
                    if model.get('name') == self.model_name:
                        logger.info(f"Model {self.model_name} is available locally")
                        self.model_loaded = True
                        return True
            
            logger.info(f"Model {self.model_name} not found locally")
            return False
            
        except Exception as e:
            logger.error(f"Failed to check model availability: {e}")
            return False
    
    def generate_text(self, prompt: str, system_prompt: str = None, 
                     temperature: float = 0.7, max_tokens: int = 1024) -> Dict[str, Any]:
        """Generate text response using Ollama model"""
        if not self.model_loaded:
            if not self.check_model_available():
                logger.warning("Model not loaded. Attempting to pull...")
                self.pull_model()
        
        request_data = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        if system_prompt:
            request_data["system"] = system_prompt
        
        try:
            response = requests.post(f'{self.base_url}/generate', json=request_data)
            if response.status_code == 200:
                result = response.json()
                return {
                    'response': result.get('response', ''),
                    'model': result.get('model', ''),
                    'total_duration': result.get('total_duration', 0),
                    'prompt_eval_count': result.get('prompt_eval_count', 0),
                    'eval_count': result.get('eval_count', 0)
                }
            else:
                logger.error(f"Error generating response: {response.status_code} - {response.text}")
                return {'error': response.text}
                
        except Exception as e:
            logger.error(f"Failed to generate text: {e}")
            return {'error': str(e)}
    
    def generate_embeddings(self, text: str) -> Dict[str, Any]:
        """Generate embeddings for text using Ollama model"""
        if not self.model_loaded:
            if not self.check_model_available():
                logger.warning("Model not loaded. Attempting to pull...")
                self.pull_model()
        
        try:
            response = requests.post(
                f'{self.base_url}/embeddings',
                json={"model": self.model_name, "prompt": text}
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'embedding': result.get('embedding', []),
                    'dim': len(result.get('embedding', [])),
                }
            else:
                logger.error(f"Error generating embeddings: {response.status_code} - {response.text}")
                return {'error': response.text}
                
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return {'error': str(e)}
    
    def process_image_text(self, image_path: str, prompt: str, 
                          temperature: float = 0.7, max_tokens: int = 1024) -> Dict[str, Any]:
        """Process image and text together using Ollama multimodal capabilities"""
        if not self.model_loaded:
            if not self.check_model_available():
                logger.warning("Model not loaded. Attempting to pull...")
                self.pull_model()
        
        try:
            # Read and encode image
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")
            
            # Prepare request with image
            request_data = {
                "model": self.model_name,
                "prompt": prompt,
                "images": [image_data],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False
            }
            
            response = requests.post(f'{self.base_url}/generate', json=request_data)
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'response': result.get('response', ''),
                    'model': result.get('model', ''),
                    'total_duration': result.get('total_duration', 0),
                    'prompt_eval_count': result.get('prompt_eval_count', 0),
                    'eval_count': result.get('eval_count', 0)
                }
            else:
                logger.error(f"Error processing image: {response.status_code} - {response.text}")
                return {'error': response.text}
                
        except Exception as e:
            logger.error(f"Failed to process image: {e}")
            return {'error': str(e)}


class QwenMultimodalManager:
    """Manager for Qwen 2.5 VL with multimodal capabilities"""
    
    def __init__(self, ollama_manager: Optional[OllamaManager] = None):
        self.ollama = ollama_manager or OllamaManager()
        self.processed_cache = {}  # Cache for processed documents
        
    def setup(self) -> bool:
        """Setup and verify environment for Qwen multimodal operations"""
        
        # Check/install Ollama
        ollama_installed = self.ollama.install_ollama()
        if not ollama_installed:
            logger.error("Failed to install Ollama")
            return False
        
        # Check/start Ollama service
        service_running = self.ollama.check_ollama_service()
        if not service_running:
            logger.error("Failed to start Ollama service")
            return False
        
        # Check/pull model
        model_ready = self.ollama.check_model_available()
        if not model_ready:
            logger.info("Model not available locally, pulling...")
            model_ready = self.ollama.pull_model()
            
        if not model_ready:
            logger.error("Failed to pull model")
            return False
            
        logger.info("Qwen Multimodal environment setup complete")
        return True
    
    def process_document_content(self, content_item, prompt_template: str = None) -> Dict[str, Any]:
        """
        Process individual document content item (text or image)
        
        Args:
            content_item: ExtractedContent item from document processor
            prompt_template: Optional template for customizing prompts
            
        Returns:
            Dict with processed results and embeddings
        """
        from src.document_processor import ContentType
        
        # Generate unique cache key
        cache_key = f"{content_item.content_id}"
        
        # Check cache first
        if cache_key in self.processed_cache:
            logger.debug(f"Using cached processing for {cache_key}")
            return self.processed_cache[cache_key]
            
        result = {
            'content_id': content_item.content_id,
            'content_type': content_item.content_type.value,
            'source_file': content_item.source_file,
            'source_page': content_item.source_page,
            'embedding': None,
            'processed_text': None,
            'error': None
        }
        
        try:
            # Process based on content type
            if content_item.content_type == ContentType.TEXT:
                # Get text embedding
                text_data = content_item.content_data
                
                # Truncate if too long
                if len(text_data) > 8000:  # Adjust based on model limits
                    logger.warning(f"Truncating long text ({len(text_data)} chars) for {content_item.content_id}")
                    text_data = text_data[:8000]
                
                # Get embeddings
                embedding_result = self.ollama.generate_embeddings(text_data)
                
                if 'error' in embedding_result:
                    result['error'] = embedding_result['error']
                else:
                    result['embedding'] = embedding_result['embedding']
                    result['processed_text'] = text_data
                
            elif content_item.content_type == ContentType.IMAGE:
                # Save image temporarily
                image = content_item.content_data
                temp_image_path = CACHE_DIR / f"temp_img_{content_item.content_id}.png"
                image.save(temp_image_path)
                
                # Process with default or custom prompt
                if prompt_template:
                    prompt = prompt_template
                else:
                    prompt = "Describe this image in detail and extract any text content visible in it."
                
                # Get image analysis
                img_result = self.ollama.process_image_text(str(temp_image_path), prompt)
                
                if 'error' in img_result:
                    result['error'] = img_result['error']
                else:
                    # Use the text description for embedding
                    text_description = img_result['response']
                    result['processed_text'] = text_description
                    
                    # Get embedding for the description
                    embedding_result = self.ollama.generate_embeddings(text_description)
                    if 'error' not in embedding_result:
                        result['embedding'] = embedding_result['embedding']
                
                # Clean up temp file
                if temp_image_path.exists():
                    temp_image_path.unlink()
                    
            else:
                result['error'] = f"Unsupported content type: {content_item.content_type.value}"
        
        except Exception as e:
            logger.error(f"Error processing content {content_item.content_id}: {e}")
            result['error'] = str(e)
        
        # Cache the result
        self.processed_cache[cache_key] = result
        
        return result
    
    def process_document(self, processed_doc) -> List[Dict[str, Any]]:
        """
        Process a complete document and generate embeddings for all content
        
        Args:
            processed_doc: ProcessedDocument from document processor
            
        Returns:
            List of dictionaries with embeddings and processed content
        """
        results = []
        
        logger.info(f"Processing document: {processed_doc.filename} " 
                  f"({len(processed_doc.extracted_contents)} content items)")
        
        # Create custom prompts based on document type
        prompt_templates = {
            'tourism': "This is from a tourism document. Describe this image in detail, identify landmarks, and extract any text content visible in it.",
            'technical': "This is from a technical manual. Describe this diagram or image in detail, identify parts or components, and extract any text content visible in it.",
            'legal': "This is from a legal document. Describe this image in detail and extract any text content visible in it.",
            'policy': "This is from a policy document. Describe this image in detail and extract any text content visible in it."
        }
        
        # Get appropriate prompt template
        doc_type = processed_doc.document_type.value
        prompt_template = prompt_templates.get(doc_type, None)
        
        # Process each content item
        for content_item in processed_doc.extracted_contents:
            result = self.process_document_content(content_item, prompt_template)
            results.append(result)
            
            # Log progress
            if result['error']:
                logger.warning(f"Error processing {content_item.content_id}: {result['error']}")
            else:
                logger.debug(f"Successfully processed {content_item.content_id}")
        
        logger.info(f"Completed processing {processed_doc.filename}: "
                  f"{len([r for r in results if not r['error']])} successful, "
                  f"{len([r for r in results if r['error']])} failed")
                  
        return results
    
    def create_multimodal_embeddings(self, processed_doc) -> Dict[str, Any]:
        """
        Create multimodal embeddings for a processed document
        
        Args:
            processed_doc: ProcessedDocument from document processor
            
        Returns:
            Dict with document metadata and embeddings
        """
        # Process all content in the document
        processed_results = self.process_document(processed_doc)
        
        # Filter out failed items
        successful_results = [r for r in processed_results if not r['error']]
        
        # Create document-level metadata
        document_data = {
            'filename': processed_doc.filename,
            'file_format': processed_doc.file_format.value,
            'document_type': processed_doc.document_type.value,
            'metadata': {
                'page_count': processed_doc.document_metadata.page_count,
                'has_images': processed_doc.document_metadata.has_images,
                'has_tables': processed_doc.document_metadata.has_tables,
                'language': processed_doc.document_metadata.language,
                'confidence_score': processed_doc.document_metadata.confidence_score
            },
            'content_embeddings': successful_results,
            'processing_stats': {
                'total_content_items': len(processed_doc.extracted_contents),
                'successfully_processed': len(successful_results),
                'failed_items': len(processed_results) - len(successful_results)
            }
        }
        
        return document_data
    
    def save_document_embeddings(self, document_data: Dict[str, Any], 
                               output_dir: Path = None) -> Path:
        """
        Save document embeddings to JSON file
        
        Args:
            document_data: Document data with embeddings
            output_dir: Directory to save the file (defaults to PROCESSED_DOCS_DIR)
            
        Returns:
            Path to saved file
        """
        output_dir = output_dir or PROCESSED_DOCS_DIR
        os.makedirs(output_dir, exist_ok=True)
        
        # Create output filename
        base_name = Path(document_data['filename']).stem
        output_file = output_dir / f"{base_name}_embeddings.json"
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(document_data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Saved document embeddings to {output_file}")
        return output_file


def test_ollama_setup():
    """Test Ollama setup and model loading"""
    print("\n🔍 Testing Ollama Setup...")
    
    # Initialize manager
    ollama_manager = OllamaManager()
    
    # Check/install Ollama
    ollama_installed = ollama_manager.install_ollama()
    print(f"✅ Ollama installed: {ollama_installed}")
    
    # Check/start service
    service_running = ollama_manager.check_ollama_service()
    print(f"✅ Ollama service running: {service_running}")
    
    # Check model availability
    model_available = ollama_manager.check_model_available()
    print(f"✅ Model available: {model_available}")
    
    if not model_available:
        print("⚠️  Model not available. Pulling now (this may take a while)...")
        model_pulled = ollama_manager.pull_model()
        print(f"✅ Model pulled: {model_pulled}")
    
    # Test simple generation
    if ollama_manager.model_loaded:
        print("\n🔍 Testing text generation...")
        response = ollama_manager.generate_text(
            "What are the top tourist attractions in Singapore?",
            max_tokens=100
        )
        
        if 'error' in response:
            print(f"❌ Generation failed: {response['error']}")
        else:
            print(f"✅ Generation successful!")
            print(f"Response: {response['response'][:150]}...")
            print(f"Duration: {response['total_duration'] / 1e9:.2f} seconds")
    
    return ollama_manager.model_loaded


def test_multimodal_manager():
    """Test the QwenMultimodalManager with a sample document"""
    from src.document_processor import DocumentProcessor
    
    print("\n🔍 Testing Multimodal Manager...")
    
    # Initialize processors
    document_processor = DocumentProcessor()
    multimodal_manager = QwenMultimodalManager()
    
    # Setup environment
    setup_success = multimodal_manager.setup()
    print(f"✅ Environment setup: {setup_success}")
    
    if not setup_success:
        print("❌ Setup failed. Cannot continue testing.")
        return False
    
    # Process sample document
    sample_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.pdf')]
    
    if not sample_files:
        print("❌ No PDF files found in data directory.")
        return False
    
    sample_file = sample_files[0]
    sample_path = DATA_DIR / sample_file
    
    print(f"📄 Processing sample document: {sample_file}")
    
    try:
        # Process document
        processed_doc = document_processor.process_document(str(sample_path))
        print(f"✅ Document processed: {len(processed_doc.extracted_contents)} content items")
        
        # Process first content item only (for testing)
        content_item = processed_doc.extracted_contents[0]
        print(f"🔍 Testing with content item: {content_item.content_id} (type: {content_item.content_type.value})")
        
        result = multimodal_manager.process_document_content(content_item)
        
        if 'error' in result and result['error']:
            print(f"❌ Processing failed: {result['error']}")
        else:
            print(f"✅ Processing successful!")
            print(f"Embedding dimension: {len(result['embedding'])}")
            
            if result['processed_text']:
                print(f"Processed text preview: {result['processed_text'][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    # Run basic tests
    test_ollama_setup()
    test_multimodal_manager()