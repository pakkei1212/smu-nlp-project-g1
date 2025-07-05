# main.py
"""
Main execution script for Multimodal RAG with Qwen2.5 VL
"""

import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from setup_directories import setup_project_structure, validate_project_setup

def main():
    """Main execution function"""
    print("🚀 Multimodal RAG with Qwen2.5 VL")
    print("=" * 50)
    
    # Setup directories and validate
    setup_project_structure()
    
    # Validate the setup
    if validate_project_setup():
        print("\n📋 Setup complete! Ready for document processing.")
        
        # Show next steps
        print("\n🎯 Next steps:")
        print("1. Place your PDF files in the 'data' folder")
        print("2. Run: python -c 'from src.document_processor import test_pdf_processing; test_pdf_processing()'")
        print("3. Continue with document processing pipeline")
    else:
        print("\n❌ Setup validation failed. Please check the errors above.")

if __name__ == "__main__":
    main()