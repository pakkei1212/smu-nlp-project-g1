
# src/setup_directories.py
import sys
import json
from pathlib import Path

# Add the project root to Python path for config import
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import *

def setup_project_structure():
    """Create the complete project directory structure"""
    
    directories = [
        DATA_DIR,
        OUTPUT_DIR, 
        CACHE_DIR,
        MODEL_CACHE_DIR,
        VECTOR_DB_PATH,
        LANGUAGE_DICT_DIR,
        PROCESSED_DOCS_DIR,
        EXTRACTED_IMAGES_DIR,
        METADATA_DIR,
        LOG_DIR
    ]
    
    print("🏗️  Setting up project directory structure...")
    print(f"📁 Base directory: {BASE_DIR}")
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory.relative_to(BASE_DIR)}")
    
    # Create sample language dictionaries
    create_sample_dictionaries()
    
    # Create __init__.py for src package
    src_init = BASE_DIR / "src" / "__init__.py"
    if not src_init.exists():
        src_init.write_text('"""Multimodal RAG source package"""\n')
        print(f"✅ Created: {src_init.relative_to(BASE_DIR)}")
    
    print(f"\n🎉 Project structure ready!")
    print(f"📋 Next steps:")
    print(f"   1. Place PDF files in: {DATA_DIR.relative_to(BASE_DIR)}")
    print(f"   2. Review language dictionaries in: {LANGUAGE_DICT_DIR.relative_to(BASE_DIR)}")
    print(f"   3. Run document processing pipeline")

def create_sample_dictionaries():
    """Create sample language dictionaries with Singapore terms"""
    
    # Singapore terms dictionary
    singapore_dict = {
        "metadata": {
            "version": "1.0",
            "description": "Singapore local terms dictionary",
            "last_updated": "2024-01-01",
            "source": "CS614 Group Project"
        },
        "languages": {
            "malay": {
                "places": {
                    "kampong_glam": {
                        "primary": "Kampong Glam",
                        "alternatives": ["Kampung Glam", "Kampong Gelam"],
                        "type": "district",
                        "description": "Historic Malay-Arab quarter",
                        "context": ["heritage", "cultural", "mosque", "arab_street"]
                    },
                    "bugis": {
                        "primary": "Bugis",
                        "alternatives": ["Bugis Street", "Bugis Junction"],
                        "type": "district",
                        "description": "Shopping and entertainment district",
                        "context": ["shopping", "entertainment", "mrt"]
                    }
                },
                "cultural_terms": {
                    "trishaw": {
                        "primary": "trishaw",
                        "alternatives": ["tri-shaw", "beca"],
                        "type": "transport",
                        "description": "Three-wheeled bicycle taxi",
                        "context": ["transport", "tourism", "traditional"]
                    }
                }
            },
            "tamil": {
                "places": {
                    "sri_veeramakaliamman": {
                        "primary": "Sri Veeramakaliamman Temple",
                        "alternatives": ["Sri Veeramakaliamman", "Veeramakaliamman Temple"],
                        "type": "temple",
                        "description": "Hindu temple in Little India",
                        "context": ["religious", "hindu", "little_india"]
                    },
                    "little_india": {
                        "primary": "Little India",
                        "alternatives": ["Little India District", "Serangoon Road"],
                        "type": "district",
                        "description": "Tamil cultural district",
                        "context": ["cultural", "tamil", "indian", "heritage"]
                    }
                }
            },
            "chinese": {
                "places": {
                    "thian_hock_keng": {
                        "primary": "Thian Hock Keng Temple",
                        "alternatives": ["天福宫", "Thian Hock Keng", "Tian Fu Gong"],
                        "type": "temple",
                        "description": "Oldest Chinese temple in Singapore",
                        "context": ["religious", "chinese", "heritage", "chinatown"]
                    },
                    "chinatown": {
                        "primary": "Chinatown",
                        "alternatives": ["China Town", "牛车水", "Niu Che Shui"],
                        "type": "district",
                        "description": "Historic Chinese district",
                        "context": ["chinese", "heritage", "cultural", "food"]
                    }
                },
                "food": {
                    "bak_kut_teh": {
                        "primary": "Bak Kut Teh",
                        "alternatives": ["肉骨茶", "bak kut teh", "pork rib soup"],
                        "type": "food",
                        "description": "Pork rib soup dish",
                        "context": ["food", "chinese", "local_cuisine"]
                    }
                }
            },
            "english": {
                "places": {
                    "marina_bay": {
                        "primary": "Marina Bay",
                        "alternatives": ["Marina Bay Sands", "MBS"],
                        "type": "district",
                        "description": "Modern waterfront district",
                        "context": ["modern", "tourism", "casino", "skyline"]
                    },
                    "sentosa": {
                        "primary": "Sentosa",
                        "alternatives": ["Sentosa Island", "Resort World Sentosa"],
                        "type": "island",
                        "description": "Resort island destination",
                        "context": ["tourism", "beaches", "theme_parks", "resort"]
                    }
                }
            }
        }
    }
    
    # Save Singapore dictionary
    singapore_file = LANGUAGE_DICT_DIR / "singapore_terms.json"
    with open(singapore_file, 'w', encoding='utf-8') as f:
        json.dump(singapore_dict, f, indent=2, ensure_ascii=False)
    print(f"✅ Created: {singapore_file.relative_to(BASE_DIR)}")
    
    # Create technical terms dictionary (for technical documents)
    technical_dict = {
        "metadata": {
            "version": "1.0", 
            "description": "Technical assembly terms dictionary",
            "last_updated": "2024-01-01",
            "source": "CS614 Group Project"
        },
        "categories": {
            "assembly_terms": {
                "cement": {
                    "primary": "cement",
                    "alternatives": ["glue", "adhesive", "bond"],
                    "type": "action",
                    "description": "To attach parts permanently",
                    "context": ["assembly", "permanent", "adhesive"]
                },
                "locate": {
                    "primary": "locate",
                    "alternatives": ["position", "place", "fit"],
                    "type": "action", 
                    "description": "To position a part in the correct place",
                    "context": ["assembly", "positioning", "placement"]
                }
            },
            "parts": {
                "chassis": {
                    "primary": "chassis",
                    "alternatives": ["frame", "body frame", "main frame"],
                    "type": "component",
                    "description": "Main structural frame",
                    "context": ["structure", "frame", "foundation"]
                }
            }
        }
    }
    
    # Save technical dictionary
    technical_file = LANGUAGE_DICT_DIR / "technical_terms.json"
    with open(technical_file, 'w', encoding='utf-8') as f:
        json.dump(technical_dict, f, indent=2, ensure_ascii=False)
    print(f"✅ Created: {technical_file.relative_to(BASE_DIR)}")

def validate_project_setup():
    """Validate that the project is set up correctly"""
    
    print("\n🔍 Validating project setup...")
    
    # Check required directories
    required_dirs = [DATA_DIR, CACHE_DIR, OUTPUT_DIR, LOG_DIR]
    for directory in required_dirs:
        if directory.exists():
            print(f"✅ {directory.relative_to(BASE_DIR)} exists")
        else:
            print(f"❌ {directory.relative_to(BASE_DIR)} missing")
            return False
    
    # Check configuration
    try:
        from config import MODEL_NAME, CHUNK_SIZE, MAX_IMAGE_SIZE
        print(f"✅ Configuration loaded successfully")
        print(f"   Model: {MODEL_NAME}")
        print(f"   Chunk size: {CHUNK_SIZE}")
        print(f"   Max image size: {MAX_IMAGE_SIZE}")
    except ImportError as e:
        print(f"❌ Configuration error: {e}")
        return False
    
    # Check language dictionaries
    dict_files = list(LANGUAGE_DICT_DIR.glob("*.json"))
    if dict_files:
        print(f"✅ Found {len(dict_files)} language dictionary files")
        for dict_file in dict_files:
            print(f"   📖 {dict_file.name}")
    else:
        print(f"⚠️  No language dictionary files found")
    
    print(f"🎉 Project validation complete!")
    return True

if __name__ == "__main__":
    setup_project_structure()
    validate_project_setup()

