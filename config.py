import os
import torch
import logging

# Logging setup for general use
LOG_DIR = "logs"

# Create logs directory before initializing FileHandler
try:
    os.makedirs(LOG_DIR, exist_ok=True)
except Exception as e:
    logging.error(f"Failed to create logs directory {LOG_DIR}: {str(e)}")
    raise

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

# Logger for chunking
CHUNK_LOG_FILE = os.path.join(LOG_DIR, "chunk_logs.log")
chunk_logger = logging.getLogger("chunk")
chunk_logger.setLevel(logging.INFO)
try:
    chunk_handler = logging.FileHandler(CHUNK_LOG_FILE)
    chunk_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    chunk_logger.addHandler(chunk_handler)
    chunk_logger.addHandler(logging.StreamHandler())  # Also output to console
except Exception as e:
    logging.error(f"Failed to initialize FileHandler for {CHUNK_LOG_FILE}: {str(e)}")
    raise

# Paths
FOLDER_PATH = "documents/output_txt"
try:
    os.makedirs(FOLDER_PATH, exist_ok=True)
    logging.info(f"Created/Verified folder: {FOLDER_PATH}")
except Exception as e:
    logging.error(f"Failed to create folder {FOLDER_PATH}: {str(e)}")
    raise

MODEL_PATH = "E:/VNPT/LLM quantize/models/qwen2.5-7b-gguf/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
# MODEL_PATH = "E:/VNPT/testChunk/models/ggml-vistral-7B-chat-q4_0.gguf"
CHROMA_DB_PATH = "chroma_test"
LOG_FILE = os.path.join(LOG_DIR, "chat_logs.json")
LLM_ANALYSIS_LOG_FILE = os.path.join(LOG_DIR, "llm_analysis_logs.json")

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Models
EMBEDDING_MODEL = "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"
SUMMARIZER_MODEL = "ntkhoi/mt5-vi-news-summarization"
RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

# Section categories
SECTION_CATEGORIES = ["triệu chứng", "nguyên nhân", "chẩn đoán", "điều trị", "phòng ngừa", "khác"]

# ChromaDB config
CHROMA_COLLECTION_NAME = "medical_docs"
CHROMA_METADATA = {
    "hnsw:space": "cosine",
    "hnsw:M": 32
}

# LLM config
LLM_CONFIG = {
    "n_gpu_layers": 30,
    "n_batch": 128,
    "n_ctx": 2048,
    "verbose": True
}

# Cấu hình OpenAI hoặc Qwen API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Thay 'os.getenv("OPENAI_API_KEY")' bằng key của bạn
QWEN_API_KEY = ""
GPT_EMBEDDING_MODEL = "text-embedding-3-small"