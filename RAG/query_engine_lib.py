# query_engine_lib.py

import torch
import transformers
import gc
import os
import psutil
import logging
import pandas as pd
from pathlib import Path

from src.model_manager import OllamaManager
from src.embedding_manager import EmbeddingManager
from src.chroma_manager import ChromaManager
from src.rag_query import RAGQueryEngine
from config import VECTOR_DB_PATH


# === ENVIRONMENT SETUP ===
def check_environment():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")

    print(f"Transformers version: {transformers.__version__}")
    try:
        _ = transformers.AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        print("✅ Successfully loaded a test model")
    except Exception as e:
        print(f"❌ Error loading test model: {e}")
    print("\nEnvironment setup completed.")


# === MEMORY & CPU MANAGEMENT ===
def setup_cpu_optimization():
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    torch.backends.quantized.engine = 'fbgemm'
    print("✅ CPU optimization configured")

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("🧹 Memory cleared")

def check_memory_usage():
    memory_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    print(f"📊 Memory Usage: {memory_mb:.1f} MB")
    return memory_mb


# === RAG PIPELINE SETUP ===
def initialize_rag_pipeline(model_name="qwen2.5vl:3b", embedding_model="nomic-embed-text",
                            collection_name="sg_explorer_documents"):
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Model Manager
    ollama_manager = OllamaManager(model_name=model_name)
    if not ollama_manager.check_model_available():
        logger.info(f"Model {model_name} not found. Pulling...")
        ollama_manager.pull_model()

    # Embedding Manager
    embedding_manager = EmbeddingManager(
        text_embedding_model=embedding_model,
        vision_model=model_name
    )

    # Chroma Manager
    chroma_manager = ChromaManager(
        persist_directory=VECTOR_DB_PATH,
        embedding_model=embedding_model,
        collection_name=collection_name
    )

    # RAG Engine
    rag_engine = RAGQueryEngine(
        embedding_manager=embedding_manager,
        chroma_manager=chroma_manager,
        ollama_manager=ollama_manager,
        default_results=3
    )

    print("✅ RAG pipeline initialized")
    return rag_engine


# === HIGH-LEVEL QUERY FUNCTION ===
def run_query(prompt, rag_engine):
    result = rag_engine.query(prompt)
    if 'error' in result:
        print(f"❌ Query failed: {result['error']}")
    else:
        print(f"✅ Query successful\n\n{result['answer']}")
    return result


# === CHROMA DB INSPECTION FUNCTIONS ===
def print_collection_stats(chroma_manager):
    stats = chroma_manager.get_collection_stats()
    print("📊 ChromaDB Collection Stats:")
    for key, value in stats.items():
        print(f"  - {key}: {value}")
    return stats

def peek_sample_items(chroma_manager, limit=10, show_documents=False):
    sample = chroma_manager.collection.peek(limit=limit)
    print(f"\n📄 Sampled {len(sample['ids'])} items:")
    for i in range(len(sample['ids'])):
        print(f"\n▶️ ID: {sample['ids'][i]}")
        print(f"   Metadata: {sample['metadatas'][i]}")
        if show_documents and "documents" in sample:
            print(f"   Document: {sample['documents'][i]}")
    return sample

def sample_to_dataframe(sample, include_documents=False):
    data = {
        "ID": sample["ids"],
        "Metadata": sample["metadatas"]
    }
    if include_documents and "documents" in sample:
        data["Document"] = sample["documents"]
    df = pd.DataFrame(data)
    return df


# === MAIN USAGE ===
if __name__ == "__main__":
    check_environment()
    setup_cpu_optimization()
    clear_memory()
    check_memory_usage()

    rag_engine = initialize_rag_pipeline()

    test_prompt = "What is the purpose of this document? Please summarise in no more than 100 words."
    run_query(test_prompt, rag_engine)

    # Optional: Inspect collection
    chroma_manager = rag_engine.chroma_manager
    print_collection_stats(chroma_manager)
    sample = peek_sample_items(chroma_manager, limit=4, show_documents=True)
    df_sample = sample_to_dataframe(sample, include_documents=True)
    print(df_sample.head())
