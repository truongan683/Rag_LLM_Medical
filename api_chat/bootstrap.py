import os
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
import openai
import logging
from modules.llm_utils import OpenAIChatWrapper

from config import (
    MODEL_PATH,
    CHROMA_DB_PATH,
    CHROMA_COLLECTION_NAME,
    LLM_CONFIG,
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    RERANKER_MODEL,
)

# ---------- LLM ----------
def init_llm():
    llm_type = "openai" # hoặc "llama"

    if llm_type == "openai":
        client = OpenAIChatWrapper(model="gpt-4o")
        return {
            "type": "openai",
            "client": client
        }

    elif llm_type == "llama":
        from llama_cpp import Llama
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=LLM_CONFIG.get("n_ctx", 2048),
            n_batch=LLM_CONFIG.get("n_batch", 128),
            n_gpu_layers=LLM_CONFIG.get("n_gpu_layers", 30),
            verbose=LLM_CONFIG.get("verbose", False)
        )
        return {
            "type": "llama",
            "client": llm
        }

    else:
        raise ValueError(f"Unsupported LLM_TYPE: {llm_type}")
# ---------- EMBEDDER ----------
def init_embedder():
    try:
        model = SentenceTransformer(EMBEDDING_MODEL)
        logging.info(f"Khởi tạo embedder thành công: {EMBEDDING_MODEL}")
        return model
    except Exception as e:
        logging.error(f"Khởi tạo embedder thất bại: {str(e)}")
        raise

# ---------- RERANKER ----------
def init_reranker():
    # Có thể tích hợp CrossEncoder nếu cần
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANKER_MODEL)
    # return None  # Bỏ qua nếu không dùng

# ---------- CHROMA CLIENT ----------
def get_chroma_client():
    try:
        os.makedirs(CHROMA_DB_PATH, exist_ok=True)
        return PersistentClient(path=CHROMA_DB_PATH)
    except Exception as e:
        logging.error(f"Không thể tạo hoặc kết nối ChromaDB tại {CHROMA_DB_PATH}: {str(e)}")
        raise

# ---------- LOAD TẤT CẢ ----------
def load_resources():
    llm = init_llm()
    client = get_chroma_client()
    collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
    embedder = init_embedder()
    reranker = init_reranker()
    return llm, collection, embedder, reranker
