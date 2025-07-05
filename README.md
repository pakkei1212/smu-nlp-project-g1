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
  - **Dataset**: `NLP\intent_classification\intent_classification_training.csv`
  - **Model**: https://drive.google.com/drive/folders/1qAycVI-8negnN62bry1ehAswjG8uKGIp?usp=sharing
- **Named Entity Recognition (NER)**: Extract locations, dates, and activities
- **Persona classification**: Infer user type for personalization
  - **Dataset**:
  1. `\NLP\persona_classification\persona_data\persona_seeds.csv`
  2. `\NLP\persona_classification\persona_data\final_train_dataset.csv`
  - **Model**: https://drive.google.com/drive/folders/1aHXhJWAQ7qeeONEd92GfGYVB8KrVF1gd?usp=sharing

### RAG/
- Uses **Qwen2.5 VL 3B** and **nomic-text-embed** to process multimodal documents (e.g., PDF with text + images)
- Performs:
  - Document chunking
  - Embedding
  - Contextual retrieval
  - Response generation using `query_engine_lib.py`
- **Dataset**:
  1. `\RAG\data\singapore_explorer_guide_image.pdf`
  2. `\RAG\data\singapore_explorer_guide_text.pdf`

### Evaluation/
- Evaluates model outputs using:
  - LLM-as-a-Judge comparisons (e.g., Gemini, GPT-4)
  - Language model perplexity
  - Output formatting and analysis
- **Dataset**: `\Evaluation\query_responses_persona.csv`

## Main_Demo.ipynb

Unified notebook that stitches together the NLP → RAG.

**Features:**
- Accepts a free-text travel query
- Extracts structured constraints and persona
- Runs RAG to generate personalized itinerary
- Dialogue management with frontend UI
- Displays JSON output with citation and explanation

You can also run:
- `Evaluation/LLM-as-a-judge.ipynb`
- `Evaluation/Perplexity.ipynb`
- `RAG/CS605_Qwen2.5VL_draft_with_image.ipynb`
