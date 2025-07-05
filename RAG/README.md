# CS605_Qwen_PAT_Shared
Repository for sharing code for RAG using Qwen_2.5 on multi-modal documents by howchih.lee.2024@mitb.smu.edu.sg and xingze.su.2024@mitb.smu.edu.sg

## Instructions
1) Create a venv with Python 3.11.9
2) Load requirements.txt in Terminal using: pip install -r requirements.txt
3) All libraries used are in requirements2.txt, but there may be conflicts.  Best to load requirements.txt
4) If you do not have ollama, Qwen2.5vl:3b, nomic-text-embed loaded from ollama, you need to do that first in Terminal.  Assuming you already downloaded ollama, go to Terminal in your sub-diretory for this project and run 
"ollama pull qwen2.5-vl:3b-instruct
ollama pull nomic-embed-text:latest" 
5) Run main.py, which will set up your project file structure
6) Run src/tests.py to ensure unittests return no bugs 
7) Document type is singapore_explorer_guide_images.pdf and singapore_explorer_guide_text.pdf (some images have text inside image, so OCR may be applied to extract text)
8) Run CS605_Qwen2.5VL_draft_with_image in Notebook to interact with the RAG documents in text and image queries.
9) query_engine_lib.py contains the core retrieval and response generation logic for the RAG pipeline. It exposes reusable functions for embedding queries, retrieving relevant document chunks, and generating context-aware responses — making it a central utility for integration with other modules.

Detailed results are saved as json files in sub-directory /output

## GitHub Actions
Everytime you push changes to GitHub repo, you can run Actions end-to-end-pipeline which will activate main.py to check that the code has no bugs.
A successful run looks like this: 
![image](https://github.com/user-attachments/assets/a01d4792-d9be-44b5-9e5e-03649050b7f9)


## Process Pipeline for Multimodal RAG with Qwen2.5 VL
With Python 3.11.9 in a virtual environment, here's a high-level overview of the process to implement multimodal RAG using Qwen2.5 VL 7B for processing PDFs containing mixed content (text, images, tables, and diagrams).
## Process Overview
### 1. Environment Setup
* Python 3.11.9 virtual environment
* Required dependencies: transformers, torch, pdf processing libraries, vector database
* GPU support for faster inference (optional but recommended)
### 2. Document Processing Pipeline
* PDF ingestion and parsing from /data
* Document segmentation into multimodal chunks (preserving visual elements)
* Handling of mixed content (text, images, tables, diagrams)
### 3. Vectorization with Qwen2.5 VL and nomic-embed_text
* Loading and configuring the Qwen2.5 VL 3B model locally
* Processing document chunks to generate multimodal embeddings
* Optimizing for hardware constraints (batching, quantization if needed)
### 4. Vector Database Integration
* Storing multimodal embeddings with metadata in /vector_db for permanence
* Implementing efficient search and retrieval
* Maintaining connections between document segments
### 5. Query Processing
* Converting user queries to compatible embeddings
* Retrieving relevant document chunks
* Passing context to Qwen2.5 VL for response generation
### 6. Output Generation
* Formatting responses based on retrieved context
* Generating explanations for tables, diagrams, and images
* Citing sources from the original documents
* Saving results into json files in /output
# CS605_Qwen_PAT_Shared
