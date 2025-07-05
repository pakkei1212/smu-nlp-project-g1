# src/embedding_manager.py
"""
Embedding manager module for generating and managing embeddings
for both text and image content using Ollama models.
"""

import os
import json
import logging
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import requests
import io
from PIL import Image

# Add project root to path for config import
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from config import *

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingManager:
    """
    Manager for generating and handling embeddings for text and image content
    """
    
    def __init__(self, 
                text_embedding_model: str = "nomic-embed-text",
                vision_model: str = "qwen2.5vl:3b",
                base_url: str = "http://localhost:11434"):
        """
        Initialize the embedding manager
        
        Args:
            text_embedding_model: Model to use for text embeddings
            vision_model: Model to use for image descriptions
            base_url: Base URL for Ollama API
        """
        self.text_embedding_model = text_embedding_model
        self.vision_model = vision_model
        self.base_url = base_url
        logger.info(f"EmbeddingManager initialized with text model: {text_embedding_model}, vision model: {vision_model}")
    
    def generate_text_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for text content
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector or None if failed
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.text_embedding_model, "prompt": text}
            )
            
            if response.status_code == 200:
                result = response.json()
                embedding = result.get('embedding', [])
                
                if embedding:
                    logger.debug(f"Generated embedding with dimension: {len(embedding)}")
                    return embedding
                else:
                    logger.error(f"Empty embedding returned")
                    return None
            else:
                logger.error(f"Error generating embedding: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to generate text embedding: {e}")
            return None
        
    def generate_image_description(self, image_path: str) -> Optional[str]:
            """
            Generate description for image using vision model
        
            Args:
            image_path: Path to image file
            
            Returns:
            Text description or None if failed
            """
            try:
                # Read and encode image
                with open(image_path, "rb") as image_file:
                    image_data = base64.b64encode(image_file.read()).decode("utf-8")
            
                # Prepare request with image
                request_data = {
                    "model": self.vision_model,
                    "prompt": "Describe this image in detail, focusing on all visual elements, text, and the overall content.",
                    "images": [image_data],
                    "stream": False
                }
            
                response = requests.post(f"{self.base_url}/api/generate", json=request_data)
            
                if response.status_code == 200:
                    result = response.json()
                    description = result.get('response', '')
                
                    if description:
                        logger.debug(f"Generated image description: {len(description)} chars")
                        return description
                    else:
                        logger.error(f"Empty description returned")
                        return None
                else:
                    logger.error(f"Error generating description: {response.status_code} - {response.text}")
                    return None
                
            except Exception as e:
                logger.error(f"Failed to generate image description: {e}")
                return None
    
    def generate_embedding_for_image(self, image_path: str) -> Optional[List[float]]:
        """
        Generate embedding for an image by first creating a description, then embedding that
        
        Args:
            image_path: Path to image file
            
        Returns:
            Embedding vector or None if failed
        """
        description = self.generate_image_description(image_path)
        if description:
            return self.generate_text_embedding(description)
        return None
    
    
    