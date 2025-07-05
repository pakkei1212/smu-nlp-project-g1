<<<<<<< HEAD
# Personal AI Travel Agent

This is a modular system that combines **Natural Language Processing (NLP)**, **Retrieval-Augmented Generation (RAG)**, and **Evaluation Pipelines** to simulate an Personal AI travel assistant. It can interpret user requirements, retrieve context from multimodal documents, generate personalized itineraries, and evaluate the outputs.

## Project Structure

```
.
├── NLP/                    # NLP module for intent classification, NER, persona detection
├── RAG/                    # RAG module for context retrieval and response generation
├── Evaluation/             # Evaluation module for judging response quality (LLM-based + perplexity)
└── Main_Demo.ipynb         # Unified notebook demo for running the full pipeline
```

## Getting Started

### 1. Setup

Use Python 3.11.9 (recommended). From the root directory:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Module Overview

### NLP/
- **Intent classification**: Identify if user wants to plan itinerary, book hotel/flight, etc.
- **Named Entity Recognition (NER)**: Extract locations, dates, and activities
- **Persona classification**: Infer user type for personalization

### RAG/
- Uses **Qwen2.5 VL 3B** and **nomic-text-embed** to process multimodal documents (e.g., PDF with text + images)
- Performs:
  - Document chunking
  - Embedding
  - Contextual retrieval
  - Response generation using `query_engine_lib.py`

### Evaluation/
- Evaluates model outputs using:
  - LLM-as-a-Judge comparisons (e.g., Gemini, GPT-4)
  - Language model perplexity
  - Output formatting and analysis

## Main_Demo.ipynb

Unified notebook that stitches together the NLP → RAG.

**Features:**
- Accepts a free-text travel query
- Extracts structured constraints and persona
- Runs RAG to generate personalized itinerary
- Evaluates response quality
- Displays JSON output with citation and explanation

You can also run:
- `Evaluation/LLM-as-a-judge.ipynb`
- `Evaluation/Perplexity.ipynb`
- `RAG/CS605_Qwen2.5VL_draft_with_image.ipynb`
=======
# smu-nlp-project-g1
Personal AI Travel Agent
>>>>>>> c3496bd2233d3a5b7594d7f87961dd52abf0a958
