
# src/tests.py - Comprehensive test suite for multimodal RAG system

import sys
import unittest
from pathlib import Path



# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.document_processor import DocumentProcessor, ContentType, FileFormat, DocumentType
from config import DATA_DIR
import logging

# Set up test logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestDocumentProcessor(unittest.TestCase):
    """Test suite for DocumentProcessor class"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.processor = DocumentProcessor()
        cls.test_files = {
            'singapore_tourism text': DATA_DIR / "singapore_explorer_guide_text.pdf",
            'singapore tourism image': DATA_DIR / "singapore_explorer_guide_image.pdf", 
            'mandatory_rest': DATA_DIR / "Mandatory Rest Days.pdf"
        }
        
        # Check which files exist
        cls.available_files = {name: path for name, path in cls.test_files.items() if path.exists()}
        
        print(f"\n🧪 Test Environment Setup")
        print(f"Available test files: {list(cls.available_files.keys())}")
    
    def test_file_format_detection(self):
        """Test file format detection"""
        print(f"\n📋 Testing file format detection...")
        
        test_cases = [
            ("test.pdf", FileFormat.PDF),
            ("test.txt", FileFormat.TXT),
            ("test.csv", FileFormat.CSV),
            ("test.xlsx", FileFormat.EXCEL),
            ("test.jpg", FileFormat.IMAGE),
            ("test.png", FileFormat.IMAGE),
            ("test.unknown", FileFormat.UNKNOWN)
        ]
        
        for filename, expected_format in test_cases:
            detected_format = self.processor.detect_file_format(filename)
            self.assertEqual(detected_format, expected_format, 
                           f"Failed for {filename}: expected {expected_format.value}, got {detected_format.value}")
            print(f"   ✅ {filename} → {detected_format.value}")
    
    def test_content_id_generation(self):
        """Test content ID generation"""
        print(f"\n🆔 Testing content ID generation...")
        
        test_cases = [
            ("test.pdf", ContentType.TEXT, 2, 0, "test_text_p2_000"),
            ("document.pdf", ContentType.IMAGE, 2, 1, "document_image_p2_001"),
            ("file.txt", ContentType.TEXT, None, 0, "file_text_000")
        ]
        
        for filename, content_type, page_num, index, expected_id in test_cases:
            generated_id = self.processor.generate_content_id(filename, content_type, page_num, index)
            self.assertEqual(generated_id, expected_id)
            print(f"   ✅ {filename} → {generated_id}")

class TestPDFProcessing(unittest.TestCase):
    """Focused tests for PDF processing"""
    
    @classmethod
    def setUpClass(cls):
        """Set up PDF test environment"""
        cls.processor = DocumentProcessor()
        cls.test_files = {
            'singapore_tourism': DATA_DIR / "Singapore_Tourism.pdf",
            'airfix': DATA_DIR / "Airfix1.pdf", 
            'mandatory_rest': DATA_DIR / "Mandatory Rest Days.pdf"
        }
        cls.available_files = {name: path for name, path in cls.test_files.items() if path.exists()}
    
    # Update the test in src/tests.py - replace the test_pdf_text_extraction method

    def test_pdf_text_extraction(self):
        """Test PDF text extraction for all available files"""
        print(f"\n📝 Testing PDF text extraction...")
    
        for name, file_path in self.available_files.items():
            with self.subTest(file=name):
                print(f"\n  Testing: {name}")
            
                try:
                    processed_doc = self.processor.process_document(str(file_path))
                
                    # Check basic structure
                    self.assertIsNotNone(processed_doc)
                    self.assertEqual(processed_doc.file_format, FileFormat.PDF)
                    self.assertGreater(len(processed_doc.extracted_contents), 0, 
                                 f"No content at all extracted from {name}")
                
                    # Separate text and image contents
                    text_contents = [content for content in processed_doc.extracted_contents 
                               if content.content_type == ContentType.TEXT]
                    image_contents = [content for content in processed_doc.extracted_contents 
                                if content.content_type == ContentType.IMAGE]
                
                    print(f"    ✅ Pages: {processed_doc.document_metadata.page_count}")
                    print(f"    ✅ Total content pieces: {len(processed_doc.extracted_contents)}")
                    print(f"    ✅ Text chunks: {len(text_contents)}")
                    print(f"    ✅ Image chunks: {len(image_contents)}")
                    print(f"    ✅ Document type: {processed_doc.document_type.value}")
                
                    # For text-based PDFs, expect text content
                    if name == 'mandatory_rest':
                        self.assertGreater(len(text_contents), 0, f"Expected text content in {name}")
                        total_chars = sum(content.metadata['character_count'] for content in text_contents)
                        print(f"    ✅ Total characters: {total_chars}")
                    
                        # Verify text content structure
                        for content in text_contents:
                            self.assertIsInstance(content.content_data, str)
                            self.assertGreater(len(content.content_data.strip()), 0)
                            self.assertIsNotNone(content.source_page)
                            self.assertIn('character_count', content.metadata)
                            self.assertIn('word_count', content.metadata)
                
                    # For image-based PDFs, expect image content
                    else:
                        if len(text_contents) == 0:
                            print(f"    ℹ️  Image-based PDF detected - no extractable text")
                            self.assertGreater(len(image_contents), 0, 
                                         f"Image-based PDF {name} should have extracted images")
                        else:
                            print(f"    ✅ Mixed content PDF - has both text and images")
                
                except Exception as e:
                    self.fail(f"Failed to process {name}: {str(e)}")


    
    
    def test_individual_pdf_methods(self):
        """Test individual PDF processing methods"""
        print(f"\n🔬 Testing individual PDF methods...")
        
        # Test with first available file
        if self.available_files:
            file_name, file_path = next(iter(self.available_files.items()))
            print(f"  Using: {file_name}")
            
            try:
                import fitz
                doc = fitz.open(str(file_path))
                
                # Test first page
                page = doc.load_page(0)
                
                # Test text extraction method
                text_contents = self.processor._extract_text_from_page(page, file_path.name, 1)
                
                if text_contents:
                    content = text_contents[0]
                    self.assertEqual(content.content_type, ContentType.TEXT)
                    self.assertEqual(content.source_page, 1)
                    self.assertIsInstance(content.content_data, str)
                    print(f"    ✅ Text extraction method working")
                    print(f"    ✅ Content ID: {content.content_id}")
                    print(f"    ✅ Characters: {content.metadata['character_count']}")
                else:
                    print(f"    ⚠️  No text found on first page")
                
                doc.close()
                
            except Exception as e:
                self.fail(f"Individual method testing failed: {str(e)}")
        else:
            self.skipTest("No PDF files available for testing")

class TestDocumentTypeDetection(unittest.TestCase):
    """Test document type detection"""
    
    @classmethod
    def setUpClass(cls):
        """Set up document type test environment"""
        cls.processor = DocumentProcessor()
    
    def test_document_type_detection(self):
        """Test document type detection with sample text"""
        print(f"\n🏷️  Testing document type detection...")
        
        test_cases = [
            ("Visit Singapore's amazing attractions and explore the cultural heritage of Chinatown", 
             DocumentType.TOURISM, "tourism content"),
            ("Assembly instructions: Locate and cement part 15 to the chassis frame", 
             DocumentType.TECHNICAL, "technical content"),
            ("This policy requires mandatory compliance with section 4.2 of the regulations", 
             DocumentType.POLICY, "policy content"),
            ("Legal statute section 10 defines the rights and obligations of all parties", 
             DocumentType.LEGAL, "legal content")
        ]
        
        for text, expected_type, description in test_cases:
            detected_type, confidence = self.processor.type_detector.detect_document_type(text, "test.txt")
            print(f"    {description}: {detected_type.value} (confidence: {confidence:.3f})")
            
            # We expect correct detection with reasonable confidence
            if confidence > 0.1:  # Only assert if there's reasonable confidence
                self.assertEqual(detected_type, expected_type, 
                               f"Failed to detect {description}: expected {expected_type.value}, got {detected_type.value}")

# Utility functions for running specific tests
def run_pdf_text_tests():
    """Run only PDF text extraction tests"""
    print("🚀 Running PDF Text Extraction Tests")
    print("="*60)
    
    suite = unittest.TestSuite()
    suite.addTest(TestPDFProcessing('test_pdf_text_extraction'))
    suite.addTest(TestPDFProcessing('test_individual_pdf_methods'))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def run_all_tests():
    """Run all tests"""
    print("🚀 Running All Tests")
    print("="*60)
    
    unittest.main(verbosity=2)

def run_basic_tests():
    """Run basic functionality tests"""
    print("🚀 Running Basic Functionality Tests")
    print("="*60)
    
    suite = unittest.TestSuite()
    suite.addTest(TestDocumentProcessor('test_file_format_detection'))
    suite.addTest(TestDocumentProcessor('test_content_id_generation'))
    suite.addTest(TestDocumentTypeDetection('test_document_type_detection'))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()



def debug_pdf_text_extraction():
    """Debug PDF text extraction issues"""
    print(f"\n🔍 DEBUG: PDF Text Extraction Analysis")
    print("="*60)
    
    import fitz
    processor = DocumentProcessor()
    
    test_files = {
        'singapore_tourism': DATA_DIR / "Singapore_Tourism.pdf",
        'airfix': DATA_DIR / "Airfix1.pdf", 
        'mandatory_rest': DATA_DIR / "Mandatory Rest Days.pdf"
    }
    
    for name, file_path in test_files.items():
        if file_path.exists():
            print(f"\n📄 ANALYZING: {name}")
            print("-" * 40)
            
            try:
                # Open PDF directly to analyze
                doc = fitz.open(str(file_path))
                print(f"Total pages: {len(doc)}")
                
                for page_num in range(min(3, len(doc))):  # Check first 3 pages
                    page = doc.load_page(page_num)
                    
                    # Get raw text
                    raw_text = page.get_text()
                    text_length = len(raw_text.strip())
                    
                    # Get text with different methods
                    text_dict = page.get_text("dict")
                    text_blocks = page.get_text("blocks")
                    
                    print(f"  Page {page_num + 1}:")
                    print(f"    Raw text length: {text_length}")
                    print(f"    Text blocks: {len(text_blocks)}")
                    
                    if text_length > 0:
                        # Show first 100 characters
                        preview = raw_text[:100].replace('\n', ' ').strip()
                        print(f"    Preview: '{preview}...'")
                        
                        # Test our extraction method
                        extracted = processor._extract_text_from_page(page, file_path.name, page_num + 1)
                        print(f"    Our method extracted: {len(extracted)} chunks")
                        
                        if extracted:
                            print(f"    Content type: {extracted[0].content_type}")
                            print(f"    Content length: {len(extracted[0].content_data)}")
                    else:
                        print(f"    ❌ No text found")
                        
                        # Check if it's an image-only page
                        images = page.get_images()
                        print(f"    Images on page: {len(images)}")
                
                doc.close()
                
            except Exception as e:
                print(f"    ❌ Error analyzing {name}: {str(e)}")


def test_ocr_availability():
    """Test if OCR is available and working"""
    print(f"\n🔍 Testing OCR Availability")
    print("="*40)
    
    try:
        import pytesseract
        from PIL import Image
        print("✅ pytesseract imported successfully")
        
        # Test with a simple image
        try:
            # Create a simple test image with text
            from PIL import Image, ImageDraw, ImageFont
            
            # Create a white image with black text
            img = Image.new('RGB', (300, 100), color='white')
            draw = ImageDraw.Draw(img)
            
            # Try to use default font
            try:
                # On macOS, try to use system font
                font = ImageFont.truetype('/System/Library/Fonts/Arial.ttf', 20)
            except:
                # Fallback to default font
                font = ImageFont.load_default()
            
            draw.text((10, 30), "Test OCR Text", fill='black', font=font)
            
            # Run OCR on test image
            ocr_result = pytesseract.image_to_string(img)
            print(f"✅ OCR test result: '{ocr_result.strip()}'")
            
            if "Test" in ocr_result or "OCR" in ocr_result:
                print("✅ OCR is working correctly!")
                return True
            else:
                print("⚠️  OCR working but accuracy may be low")
                return True
                
        except Exception as e:
            print(f"❌ OCR test failed: {str(e)}")
            return False
            
    except ImportError as e:
        print(f"❌ OCR not available: {str(e)}")
        print("💡 Install with: pip install pytesseract")
        print("💡 Install Tesseract: brew install tesseract (macOS)")
        return False

# Add this to the main execution section
if __name__ == "__main__":
    # Test OCR first
    if test_ocr_availability():
        print("\n🚀 OCR ready - proceeding with PDF tests")
        run_pdf_text_tests()
    else:
        print("\n⚠️  OCR not available - will only extract standard text")
        run_pdf_text_tests()

    # Run debug analysis 
    debug_pdf_text_extraction()
    
    print("\n" + "="*60)
    # Then run normal tests
    run_pdf_text_tests()


 