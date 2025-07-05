# src/__init__.py
"""
Multimodal RAG with Qwen2.5 VL source package in Python 

"""


from .setup_directories import setup_project_structure
from .document_processor import DocumentProcessor

__all__ = ['setup_project_structure', 'DocumentProcessor']

__version__ = "1.0.0"
__author__ = "CS614 Group Project"