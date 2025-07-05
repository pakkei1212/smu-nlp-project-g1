# src/document_processor.py


import fitz  # PyMuPDF
import pandas as pd
import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
from PIL import Image
import io
import logging
from dataclasses import dataclass
from enum import Enum
from datetime import datetime


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    logger.warning("OCR not available - install pytesseract for image-based text extraction")

# Add project root to path for config import
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from config import *

# Hybrid Document Embedding Approach

class DocumentType(Enum):
    """Enumeration of possible document types"""
    TOURISM = "tourism"
    TECHNICAL = "technical" 
    LEGAL = "legal"
    POLICY = "policy"
    FINANCIAL = "financial"
    ACADEMIC = "academic"
    MARKETING = "marketing"
    INSTRUCTION = "instruction"
    REFERENCE = "reference"
    MIXED = "mixed"
    UNKNOWN = "unknown"

class FileFormat(Enum):
    """Primary file containers"""
    PDF = "pdf"
    TXT = "txt"
    EXCEL = "excel"
    CSV = "csv"
    IMAGE = "image"
    WORD = "word"
    UNKNOWN = "unknown"

class ContentType(Enum):
    """Types of content within files"""
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    DIAGRAM = "diagram"
    CHART = "chart"
    MIXED = "mixed"


@dataclass
class DocumentMetadata:
    """Document metadata structure"""
    filename: str
    file_format: FileFormat
    document_type: DocumentType
    confidence_score: float
    page_count: int
    has_images: bool
    has_tables: bool
    language: str
    file_size: int
    processing_timestamp: str

@dataclass
class ExtractedContent:
    """Individual piece of extracted content"""
    content_id: str          # Unique identifier
    content_type: ContentType
    source_file: str         # Original filename
    source_page: Optional[int] # Page number (for PDFs)
    content_data: Any        # Text string, Image object, or DataFrame
    position_info: Optional[Dict] # Spatial position in source
    metadata: Dict[str, Any] # Additional metadata

@dataclass
class ProcessedDocument:
    """Complete processed document structure"""
    filename: str
    file_format: FileFormat
    document_type: DocumentType
    extracted_contents: List[ExtractedContent]  # All content pieces
    document_metadata: DocumentMetadata



class DocumentTypeDetector:
    """Intelligent document type detection based on content analysis.  Can add to vocab list.
    
    """
    
    def __init__(self):
        self.type_indicators = {
            DocumentType.TOURISM: {
                'keywords': ['tourism', 'tourist', 'visit', 'explore', 'attraction', 'hotel', 'restaurant', 
                           'sightseeing', 'heritage', 'culture', 'district', 'guide', 'travel'],
                'patterns': [r'\b(visit|explore|discover)\s+\w+', r'things\s+to\s+do', r'attractions?'],
                'weight': 1.0
            },
            DocumentType.TECHNICAL: {
                'keywords': ['assembly', 'instructions', 'parts', 'components', 'step', 'procedure',
                           'installation', 'manual', 'specification', 'diagram', 'cement', 'locate'],
                'patterns': [r'step\s+\d+', r'part\s+\d+', r'assembly\s+\w+', r'locate\s+and\s+cement'],
                'weight': 1.0
            },
            DocumentType.LEGAL: {
                'keywords': ['law', 'legal', 'regulation', 'compliance', 'statute', 'act', 'section',
                           'clause', 'provision', 'rights', 'obligations', 'liability', 'contract'],
                'patterns': [r'section\s+\d+', r'clause\s+\d+', r'pursuant\s+to', r'shall\s+be'],
                'weight': 1.0
            },
            DocumentType.POLICY: {
                'keywords': ['policy', 'procedure', 'guidelines', 'rules', 'requirements', 'mandatory',
                           'compliance', 'standards', 'protocol', 'framework', 'governance'],
                'patterns': [r'policy\s+\d+', r'mandatory\s+\w+', r'shall\s+comply', r'requirements?'],
                'weight': 1.0
            }
        }
    
    def detect_document_type(self, text_content: str, filename: str):
        """
        Detect document type based on content analysis
        
        Args:
            text_content (str): Extracted text from document
            filename (str): Document filename for additional context
            
        Returns:
            Tuple[DocumentType, float]: Detected type and confidence score
        """
        import re
        
        if not text_content or len(text_content.strip()) < 50:
            return DocumentType.UNKNOWN, 0.0
        
        text_lower = text_content.lower()
        type_scores = {}
        
        # Score each document type
        for doc_type, indicators in self.type_indicators.items():
            score = 0.0
            
            # Keyword matching
            keyword_matches = sum(1 for keyword in indicators['keywords'] 
                                if keyword in text_lower)
            keyword_score = (keyword_matches / len(indicators['keywords'])) * 0.7
            
            # Pattern matching
            pattern_matches = sum(1 for pattern in indicators['patterns']
                                if re.search(pattern, text_lower))
            pattern_score = (pattern_matches / len(indicators['patterns'])) * 0.3
            
            total_score = (keyword_score + pattern_score) * indicators['weight']
            type_scores[doc_type] = total_score
        
        # Additional filename-based scoring
        filename_lower = filename.lower()
        if 'tourism' in filename_lower or 'guide' in filename_lower:
            type_scores[DocumentType.TOURISM] = type_scores.get(DocumentType.TOURISM, 0) + 0.2
        elif 'assembly' in filename_lower or 'manual' in filename_lower:
            type_scores[DocumentType.TECHNICAL] = type_scores.get(DocumentType.TECHNICAL, 0) + 0.2
        elif 'legal' in filename_lower or 'law' in filename_lower:
            type_scores[DocumentType.LEGAL] = type_scores.get(DocumentType.LEGAL, 0) + 0.2
        elif 'policy' in filename_lower or 'mandatory' in filename_lower:
            type_scores[DocumentType.POLICY] = type_scores.get(DocumentType.POLICY, 0) + 0.2
        
        # Get best match
        if not type_scores or max(type_scores.values()) < 0.1:
            return DocumentType.UNKNOWN, 0.0
        
        best_type = max(type_scores, key=type_scores.get)
        confidence = min(type_scores[best_type], 1.0)
        
        logger.info(f"Document type detection for {filename}: {best_type.value} (confidence: {confidence:.3f})")
        
        return best_type, confidence


class DocumentProcessor:
    """Enhanced document processing class for multiple file formats and document types"""
    
    def __init__(self):
        self.config = {
            'max_image_size': MAX_IMAGE_SIZE,
            'chunk_size': CHUNK_SIZE,
            'chunk_overlap': CHUNK_OVERLAP,
            'data_dir': DATA_DIR,
            'cache_dir': CACHE_DIR
        }
        self.type_detector = DocumentTypeDetector()
        logger.info(f"DocumentProcessor initialized with config: {self.config}")
    
    def detect_file_format(self, file_path: str) -> FileFormat:
        """Detect file format based on extension"""
        ext = Path(file_path).suffix.lower()
        
        format_map = {
            '.pdf': FileFormat.PDF,
            '.txt': FileFormat.TXT,
            '.xlsx': FileFormat.EXCEL,
            '.xls': FileFormat.EXCEL,
            '.csv': FileFormat.CSV,
            '.jpg': FileFormat.IMAGE,
            '.jpeg': FileFormat.IMAGE,
            '.png': FileFormat.IMAGE,
            '.gif': FileFormat.IMAGE,
            '.bmp': FileFormat.IMAGE,
            '.docx': FileFormat.WORD,
            '.doc': FileFormat.WORD
        }
        
        return format_map.get(ext, FileFormat.UNKNOWN)
    
    def generate_content_id(self, filename: str, content_type: ContentType, 
                          page_num: Optional[int] = None, index: int = 0) -> str:
        """Generate unique content ID"""
        base_name = Path(filename).stem
        page_part = f"_p{page_num}" if page_num is not None else ""
        return f"{base_name}_{content_type.value}{page_part}_{index:03d}"
    
    def process_document(self, file_path: str) -> ProcessedDocument:
        
        ## Main document function, handles multiple file formats


        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_format = self.detect_file_format(file_path)
        filename = Path(file_path).name
        file_size = os.path.getsize(file_path)
        
        logger.info(f"Processing {filename} (format: {file_format.value}, size: {file_size} bytes)")
        
        # Route to appropriate processor based on file format
        try:
            if file_format == FileFormat.PDF:
                extracted_contents, basic_metadata = self._process_pdf(file_path)
            elif file_format == FileFormat.TXT:
                extracted_contents, basic_metadata = self._process_txt(file_path)
            elif file_format == FileFormat.CSV:
                extracted_contents, basic_metadata = self._process_csv(file_path)
            elif file_format == FileFormat.EXCEL:
                extracted_contents, basic_metadata = self._process_excel(file_path)
            elif file_format == FileFormat.IMAGE:
                extracted_contents, basic_metadata = self._process_image(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_format.value}")
            
            # Detect document type based on extracted text
            all_text = " ".join([content.content_data for content in extracted_contents 
                               if content.content_type == ContentType.TEXT and 
                               isinstance(content.content_data, str)])
            
            document_type, confidence = self.type_detector.detect_document_type(all_text, filename)
            
            # Create document metadata
            from datetime import datetime
            document_metadata = DocumentMetadata(
                filename=filename,
                file_format=file_format,
                document_type=document_type,
                confidence_score=confidence,
                page_count=basic_metadata.get('page_count', 1),
                has_images=any(content.content_type == ContentType.IMAGE for content in extracted_contents),
                has_tables=any(content.content_type == ContentType.TABLE for content in extracted_contents),
                language=basic_metadata.get('language', 'en'),
                file_size=file_size,
                processing_timestamp=datetime.now().isoformat()
            )
            
            # Create processed document
            processed_doc = ProcessedDocument(
                filename=filename,
                file_format=file_format,
                document_type=document_type,
                extracted_contents=extracted_contents,
                document_metadata=document_metadata
            )
            
            logger.info(f"Successfully processed {filename}: "
                       f"{len(extracted_contents)} content pieces extracted, "
                       f"type: {document_type.value} (confidence: {confidence:.3f})")
            
            return processed_doc
            
        except Exception as e:
            logger.error(f"Failed to process {filename}: {str(e)}")
            raise
    
    def _process_txt(self, file_path: str) -> Tuple[List[ExtractedContent], Dict[str, Any]]:
        """Process TXT files"""
        filename = Path(file_path).name
        
        try:
            # Try different encodings
            encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
            text_content = None
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        text_content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if text_content is None:
                raise ValueError("Could not decode text file with any supported encoding")
            
            # Create single text content
            content = ExtractedContent(
                content_id=self.generate_content_id(filename, ContentType.TEXT),
                content_type=ContentType.TEXT,
                source_file=filename,
                source_page=None,
                content_data=text_content.strip(),
                position_info=None,
                metadata={'character_count': len(text_content), 'line_count': len(text_content.splitlines())}
            )
            
            basic_metadata = {
                'page_count': 1,
                'language': 'en',  # Could implement language detection here
                'encoding': encoding
            }
            
            return [content], basic_metadata
            
        except Exception as e:
            logger.error(f"Error processing TXT file {filename}: {str(e)}")
            raise
    
    def _process_csv(self, file_path: str) -> Tuple[List[ExtractedContent], Dict[str, Any]]:
        """Process CSV files"""
        filename = Path(file_path).name
        
        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Create table content
            content = ExtractedContent(
                content_id=self.generate_content_id(filename, ContentType.TABLE),
                content_type=ContentType.TABLE,
                source_file=filename,
                source_page=None,
                content_data=df,
                position_info=None,
                metadata={
                    'row_count': len(df),
                    'column_count': len(df.columns),
                    'columns': list(df.columns),
                    'data_types': df.dtypes.to_dict()
                }
            )
            
            basic_metadata = {
                'page_count': 1,
                'language': 'en',
                'row_count': len(df),
                'column_count': len(df.columns)
            }
            
            return [content], basic_metadata
            
        except Exception as e:
            logger.error(f"Error processing CSV file {filename}: {str(e)}")
            raise
    
    def _process_image(self, file_path: str) -> Tuple[List[ExtractedContent], Dict[str, Any]]:
        """Process standalone image files"""
        filename = Path(file_path).name
        
        try:
            # Load image
            image = Image.open(file_path)
            
            # Resize if too large
            if max(image.size) > self.config['max_image_size']:
                image.thumbnail((self.config['max_image_size'], self.config['max_image_size']), 
                              Image.Resampling.LANCZOS)
            
            # Create image content
            content = ExtractedContent(
                content_id=self.generate_content_id(filename, ContentType.IMAGE),
                content_type=ContentType.IMAGE,
                source_file=filename,
                source_page=None,
                content_data=image,
                position_info=None,
                metadata={
                    'width': image.size[0],
                    'height': image.size[1],
                    'format': image.format,
                    'mode': image.mode
                }
            )
            
            basic_metadata = {
                'page_count': 1,
                'language': 'visual',
                'image_format': image.format
            }
            
            return [content], basic_metadata
            
        except Exception as e:
            logger.error(f"Error processing image file {filename}: {str(e)}")
            raise
  

    def _process_pdf(self, file_path: str) -> Tuple[List[ExtractedContent], Dict[str, Any]]:
        """Process PDF files - extract text and images separately"""

        filename = Path(file_path).name
        extracted_contents = []
        
        try:
            # Open PDF document
            doc = fitz.open(file_path)
            total_pages = len(doc)
            total_images = 0
            total_text_length = 0
            
            logger.info(f"Processing PDF {filename} with {total_pages} pages")
            
            # Process each page
            for page_num in range(total_pages):
                page = doc.load_page(page_num)
                page_number = page_num + 1
                
                # Extract text content from this page
                text_contents = self._extract_text_from_page(page, filename, page_number)
                extracted_contents.extend(text_contents)
                
                # Extract images from this page
                image_contents = self._extract_images_from_page(page, doc, filename, page_number)
                extracted_contents.extend(image_contents)
                total_images += len(image_contents)
            
            # Close document
            doc.close()
            
            # Calculate totals
            total_text_length = sum(len(content.content_data) for content in extracted_contents 
                                  if content.content_type == ContentType.TEXT)
            
            basic_metadata = {
                'page_count': total_pages,
                'total_images': total_images,
                'total_text_length': total_text_length,
                'language': 'en'  # Could implement language detection
            }
            
            logger.info(f"PDF processing complete: {len(extracted_contents)} content pieces extracted")
            return extracted_contents, basic_metadata
            
        except Exception as e:
            logger.error(f"Error processing PDF {filename}: {str(e)}")
            raise




    def _extract_text_from_page(self, page, filename: str, page_number: int) -> List[ExtractedContent]:
        """Extract text content from a PDF page with OCR fallback"""
        text_contents = []
    
        try:
            # First, try standard text extraction methods
            extraction_methods = [
                ("default", lambda: page.get_text()),
                ("blocks", lambda: "\n".join([block[4] for block in page.get_text("blocks") if len(block) > 4])),
                ("dict", lambda: self._extract_text_from_dict(page.get_text("dict")))
            ]
        
            best_text = ""
            best_method = "none"
        
            for method_name, extraction_func in extraction_methods:
                try:
                    text_content = extraction_func()
                    if text_content and len(text_content.strip()) > len(best_text.strip()):
                        best_text = text_content
                        best_method = method_name
                        logger.debug(f"Method '{method_name}' extracted {len(text_content)} chars from page {page_number}")
                except Exception as e:
                    logger.debug(f"Method '{method_name}' failed for page {page_number}: {str(e)}")
        
            # If no text found with standard methods, try OCR
            if not best_text.strip() and OCR_AVAILABLE:
                logger.info(f"No standard text found on page {page_number}, attempting OCR...")
                ocr_text = self._extract_text_with_ocr(page, filename, page_number)
                if ocr_text.strip():
                    best_text = ocr_text
                    best_method = "ocr"
                    logger.info(f"OCR extracted {len(ocr_text)} characters from page {page_number}")
        
            # Create text content object if we found any text
            if best_text.strip():
                text_obj = ExtractedContent(
                    content_id=self.generate_content_id(filename, ContentType.TEXT, page_number),
                    content_type=ContentType.TEXT,
                    source_file=filename,
                    source_page=page_number,
                    content_data=best_text.strip(),
                    position_info={'page': page_number},
                    metadata={
                        'character_count': len(best_text),
                        'word_count': len(best_text.split()),
                        'extraction_method': best_method,
                        'is_ocr': best_method == "ocr",
                        'page_dimensions': {
                            'width': page.rect.width, 
                            'height': page.rect.height
                        }
                    }
                )
                text_contents.append(text_obj)
            
                logger.debug(f"Extracted {len(best_text)} characters from page {page_number} using {best_method}")
            else:
                logger.warning(f"No text found on page {page_number} with any method (including OCR)")
    
        except Exception as e:
            logger.warning(f"Could not extract text from page {page_number}: {str(e)}")
    
        return text_contents

    

    def _extract_text_from_dict(self, text_dict):
        """Extract text from PyMuPDF dict format"""
        text_parts = []
    
        for block in text_dict.get("blocks", []):
            if "lines" in block:  # Text block
                for line in block["lines"]:
                    for span in line.get("spans", []):
                        text_parts.append(span.get("text", ""))
    
        return " ".join(text_parts)



    def _extract_images_from_page(self, page, doc, filename: str, page_number: int) -> List[ExtractedContent]:
        """Extract images from a PDF page"""
        image_contents = []
        
        try:
            # Get list of images on this page
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                try:
                    # Get image reference
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    # Skip CMYK images (they cause conversion issues)
                    if pix.n - pix.alpha < 4:
                        # Convert to PIL Image
                        img_data = pix.tobytes("ppm")
                        pil_image = Image.open(io.BytesIO(img_data))
                        
                        # Resize if too large
                        original_size = pil_image.size
                        if max(pil_image.size) > self.config['max_image_size']:
                            pil_image.thumbnail(
                                (self.config['max_image_size'], self.config['max_image_size']), 
                                Image.Resampling.LANCZOS
                            )
                        
                        # Create image content object
                        image_obj = ExtractedContent(
                            content_id=self.generate_content_id(filename, ContentType.IMAGE, 
                                                              page_number, img_index),
                            content_type=ContentType.IMAGE,
                            source_file=filename,
                            source_page=page_number,
                            content_data=pil_image,
                            position_info={
                                'page': page_number,
                                'image_index': img_index
                            },
                            metadata={
                                'original_size': original_size,
                                'current_size': pil_image.size,
                                'format': pil_image.format,
                                'mode': pil_image.mode,
                                'resized': max(original_size) > self.config['max_image_size']
                            }
                        )
                        
                        image_contents.append(image_obj)
                        logger.debug(f"Extracted image {img_index} from page {page_number}")
                    
                    # Clean up pixmap
                    pix = None
                    
                except Exception as e:
                    logger.warning(f"Could not extract image {img_index} from page {page_number}: {str(e)}")
                    continue
        
        except Exception as e:
            logger.warning(f"Could not process images from page {page_number}: {str(e)}")
        
        return image_contents



    def _extract_text_with_ocr(self, page, filename: str, page_number: int) -> str:
        """Extract text using OCR from a PDF page"""
        
        if not OCR_AVAILABLE:
            logger.warning("OCR requested but pytesseract not available")
            return ""
        
        try:
            # Convert page to high-resolution image for better OCR
            mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better OCR accuracy
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image
            img_data = pix.tobytes("ppm")
            pil_image = Image.open(io.BytesIO(img_data))
            

            # Use default OCR settings
            ocr_text = pytesseract.image_to_string(pil_image, lang='eng')

            ## Apply OCR with optimized settings
            # custom_config = r'--oem 3 --psm 6'  
            
            ## Extract text using OCR
            # ocr_text = pytesseract.image_to_string(
            #     pil_image, 
            #     config=custom_config,
            #     lang='eng'  # Can be extended to support multiple languages
            # )
            
            # Clean up
            pix = None
            pil_image.close()
            
            # Clean OCR output
            cleaned_text = ocr_text.strip()
            
            logger.debug(f"OCR on page {page_number}: {len(ocr_text)} raw chars -> {len(cleaned_text)} cleaned chars")
            
            return cleaned_text
            
        except Exception as e:
            logger.warning(f"OCR failed on page {page_number}: {str(e)}")
            return ""

    def _clean_ocr_text(self, ocr_text: str) -> str:
        """Clean and normalize OCR output"""
        import re
        
        if not ocr_text:
            return ""
        
        # Remove excessive whitespace
        cleaned = re.sub(r'\s+', ' ', ocr_text)
        
        # Remove common OCR artifacts
        cleaned = re.sub(r'[|]{2,}', '', cleaned)  # Remove multiple pipes
        cleaned = re.sub(r'[_]{3,}', '', cleaned)  # Remove multiple underscores
        
        # Fix common OCR character errors
        replacements = {
            '|': 'I',  # Pipe to I
            '0': 'O',  # Zero to O in obvious text contexts
            '5': 'S',  # 5 to S in obvious text contexts
        }
        
        # Apply replacements only in word contexts
        for old, new in replacements.items():
            # Only replace if surrounded by letters (word context)
            cleaned = re.sub(f'(?<=[a-zA-Z]){re.escape(old)}(?=[a-zA-Z])', new, cleaned)
        
        return cleaned.strip()
    
    # Add this method to your existing DocumentProcessor class

    def process_mixed_pdf(self, text_pdf_path: str, image_pdf_path: str) -> ProcessedDocument:
        """
        Process a mixed-content PDF with flattened images using 3-prong approach with pre-created masked PDFs
    
        Args:
            text_pdf_path: Path to PDF with images masked out (text-only)
            image_pdf_path: Path to PDF with text masked out (image-only)
        
        Returns:
            ProcessedDocument: Combined document with text and image content plus coordinate mapping
        """
        logger.info(f"Processing mixed PDF: text={Path(text_pdf_path).name}, image={Path(image_pdf_path).name}")
    
        # Validate input files
        if not os.path.exists(text_pdf_path):
            raise FileNotFoundError(f"Text PDF not found: {text_pdf_path}")
        if not os.path.exists(image_pdf_path):
            raise FileNotFoundError(f"Image PDF not found: {image_pdf_path}")
    
        # Process text-only PDF
        logger.info("Processing text-only PDF...")
        text_processed_doc = self.process_document(text_pdf_path)
    
        # Process image-only PDF  
        logger.info("Processing image-only PDF...")
        image_processed_doc = self._process_pdf_pages_as_images(image_pdf_path)
    
        # Combine extracted contents
        combined_contents = []
        combined_contents.extend(text_processed_doc.extracted_contents)
        combined_contents.extend(image_processed_doc.extracted_contents)
    
        # Generate coordinate mapping
        coordinate_mapping = self._generate_coordinate_mapping(
            text_processed_doc, 
            image_processed_doc
        )
    
        # Create combined metadata
        combined_metadata = DocumentMetadata(
            filename=f"mixed_{Path(text_pdf_path).stem}",
            file_format=FileFormat.PDF,
            document_type=text_processed_doc.document_type,  # Use text PDF's detected type
            confidence_score=text_processed_doc.document_metadata.confidence_score,
            page_count=max(text_processed_doc.document_metadata.page_count, 
                      image_processed_doc.document_metadata.page_count),
            has_images=True,
            has_tables=text_processed_doc.document_metadata.has_tables,
            language=text_processed_doc.document_metadata.language,
            file_size=text_processed_doc.document_metadata.file_size + image_processed_doc.document_metadata.file_size,
            processing_timestamp=datetime.now().isoformat()
        )
    
        # Create combined processed document
        combined_doc = ProcessedDocument(
            filename=f"mixed_{Path(text_pdf_path).stem}",
            file_format=FileFormat.PDF,
            document_type=text_processed_doc.document_type,
            extracted_contents=combined_contents,
            document_metadata=combined_metadata
        )
    
        # Add coordinate mapping to metadata
        combined_doc.coordinate_mapping = coordinate_mapping
    
        logger.info(f"Mixed PDF processing complete: {len(combined_contents)} total content pieces")
        logger.info(f"  - Text items: {len([c for c in combined_contents if c.content_type == ContentType.TEXT])}")
        logger.info(f"  - Image items: {len([c for c in combined_contents if c.content_type == ContentType.IMAGE])}")
    
        return combined_doc

    def _generate_coordinate_mapping(self, text_doc: ProcessedDocument, image_doc: ProcessedDocument) -> Dict[str, Any]:
        """
        Generate coordinate mapping between text and image content
    
        Args:
            text_doc: Processed text-only document
            image_doc: Processed image-only document
        
        Returns:
            Dictionary containing coordinate mapping information
        """
        mapping = {
            "text_pdf": text_doc.filename,
            "image_pdf": image_doc.filename,
            "page_mappings": {},
            "content_relationships": []
        }
    
        # Create page-level mapping
        for page_num in range(1, max(text_doc.document_metadata.page_count, 
                                image_doc.document_metadata.page_count) + 1):
            # Get text content for this page
            text_items = [c for c in text_doc.extracted_contents 
                     if c.source_page == page_num and c.content_type == ContentType.TEXT]
        
            # Get image content for this page
            image_items = [c for c in image_doc.extracted_contents 
                      if c.source_page == page_num and c.content_type == ContentType.IMAGE]
        
            mapping["page_mappings"][str(page_num)] = {
                "text_content_ids": [item.content_id for item in text_items],
                "image_content_ids": [item.content_id for item in image_items],
                "text_count": len(text_items),
                "image_count": len(image_items)
            }
        
            # Create content relationships (same page = related)
            for text_item in text_items:
                for image_item in image_items:
                    mapping["content_relationships"].append({
                    "text_content_id": text_item.content_id,
                    "image_content_id": image_item.content_id,
                    "page": page_num,
                    "relationship_type": "same_page"
                    })
    
        logger.info(f"Generated coordinate mapping for {len(mapping['page_mappings'])} pages")
        logger.info(f"  - Created {len(mapping['content_relationships'])} content relationships")
    
        return mapping
    
    def _process_pdf_pages_as_images(self, image_pdf_path: str) -> ProcessedDocument:
        """
        Process PDF by converting each page to an image
    
        Args:
        image_pdf_path: Path to PDF file
        
        Returns:
        ProcessedDocument with page images as ExtractedContent
        """
        filename = Path(image_pdf_path).name
        extracted_contents = []
    
        try:
            # Open PDF document
            doc = fitz.open(image_pdf_path)
            total_pages = len(doc)
        
            logger.info(f"Converting {total_pages} PDF pages to images: {filename}")
        
            # Process each page as an image
            for page_num in range(total_pages):
                page = doc.load_page(page_num)
                page_number = page_num + 1
                # Convert page to high-resolution image
                mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
                pix = page.get_pixmap(matrix=mat)

                # Convert to PIL Image
                img_data = pix.tobytes("png")
                pil_image = Image.open(io.BytesIO(img_data))

                # Resize if too large
                original_size = pil_image.size
                if max(pil_image.size) > self.config['max_image_size']:
                    pil_image.thumbnail(
                    (self.config['max_image_size'], self.config['max_image_size']), 
                    Image.Resampling.LANCZOS
                )
            
                # Create image content object
                image_obj = ExtractedContent(
                    content_id=self.generate_content_id(filename, ContentType.IMAGE, page_number, 0),
                    content_type=ContentType.IMAGE,
                    source_file=filename,
                    source_page=page_number,
                    content_data=pil_image,
                    position_info={
                    'page': page_number,
                    'full_page_image': True,
                    'conversion_method': 'pdf_page_to_image'
                    },
                    metadata={
                    'original_size': original_size,
                    'current_size': pil_image.size,
                    'format': 'PNG',
                    'mode': pil_image.mode,
                    'resized': max(original_size) > self.config['max_image_size'],
                    'pdf_page_number': page_number,
                    'zoom_factor': 2.0
                    }
                )
            
                extracted_contents.append(image_obj)
                logger.debug(f"Converted page {page_number} to image: {pil_image.size}")
            
                # Clean up pixmap
                pix = None
        
            # Close document
            doc.close()
        
            # Detect document type based on filename (since no text to analyze)
            document_type, confidence = self.type_detector.detect_document_type(
                f"tourism guide {filename}", filename
            )
        
            # Create document metadata
            from datetime import datetime
            document_metadata = DocumentMetadata(
                filename=filename,
                file_format=FileFormat.PDF,
                document_type=document_type,
                confidence_score=confidence,
                page_count=total_pages,
                has_images=True,
                has_tables=False,
                language='visual',
                file_size=os.path.getsize(image_pdf_path),
                processing_timestamp=datetime.now().isoformat()
            )
        
            # Create processed document
            processed_doc = ProcessedDocument(
                filename=filename,
                file_format=FileFormat.PDF,
                document_type=document_type,
                extracted_contents=extracted_contents,
                document_metadata=document_metadata
            )
        
            logger.info(f"PDF page conversion complete: {len(extracted_contents)} page images extracted")

            return processed_doc

        except Exception as e:
            logger.error(f"Error converting PDF pages to images {filename}: {str(e)}")
            raise

