
# config.py for Qwen2.5-VL-3B-Instruct project
import os
from pathlib import Path

# Project directories
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
CACHE_DIR = BASE_DIR / "cache"

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Model configuration
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_CACHE_DIR = CACHE_DIR / "models"
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# Vector database configuration
VECTOR_DB_PATH = CACHE_DIR / "vector_db"
os.makedirs(VECTOR_DB_PATH, exist_ok=True)

# Language dictionary configuration (NEW)
LANGUAGE_DICT_DIR = CACHE_DIR / "lang_dict" # Directory for local languages or special terminology 
os.makedirs(LANGUAGE_DICT_DIR, exist_ok=True)

# Processing configuration
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 128
MAX_IMAGE_SIZE = 1024  # Maximum dimension for images

# Language processing configuration (NEW)
SUPPORTED_LANGUAGES = ["malay", "tamil", "chinese", "english"]
LANGUAGE_VALIDATION_ENABLED = True
CULTURAL_CONTEXT_ENRICHMENT = True
ALTERNATIVE_SPELLING_TOLERANCE = 0.8  # Similarity threshold for spell checking

# Document processing configuration (NEW)
SUPPORTED_PDF_TYPES = ["tourism", "technical", "mixed"]
AUTO_DETECT_DOCUMENT_TYPE = True
MIN_TEXT_QUALITY_SCORE = 0.7  # Minimum OCR quality score
MIN_IMAGE_RESOLUTION = 300  # Minimum DPI for high-quality images

# Processing paths (NEW)
PROCESSED_DOCS_DIR = CACHE_DIR / "processed_documents"
EXTRACTED_IMAGES_DIR = CACHE_DIR / "extracted_images"
METADATA_DIR = CACHE_DIR / "metadata"

# Create additional directories
os.makedirs(PROCESSED_DOCS_DIR, exist_ok=True)
os.makedirs(EXTRACTED_IMAGES_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)

# Debug and logging configuration (NEW)
LOG_DIR = BASE_DIR / "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
ENABLE_DETAILED_LOGGING = True

print(f"✅ Config loaded - Base directory: {BASE_DIR}")
print(f"✅ Data directory: {DATA_DIR}")
print(f"✅ Cache directory: {CACHE_DIR}")
print(f"✅ Language dictionaries: {LANGUAGE_DICT_DIR}")
