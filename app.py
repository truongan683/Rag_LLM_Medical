import os
import sys

# 1. Tắt Streamlit file watcher để tránh inspect các module PyTorch
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

# 2. Trên Windows: đặt event loop policy rõ ràng
if sys.platform.startswith("win"):
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# 3. Patch torch.classes để __path__ luôn là list rỗng
import torch
torch.classes.__path__ = []

# 4. Import các thành phần
import streamlit as st
from config import (
    MODEL_PATH, LLM_CONFIG, EMBEDDING_MODEL, RERANKER_MODEL, CHROMA_DB_PATH,
    OPENAI_API_KEY, GPT_EMBEDDING_MODEL
)
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
import openai
from modules.llm_utils import OpenAIChatWrapper
# Gán API key nếu dùng OpenAI
openai.api_key = OPENAI_API_KEY

def init_models(llm_choice):
    if "models_loaded" not in st.session_state:
        st.session_state["client"] = chromadb.PersistentClient(path=CHROMA_DB_PATH)

        if llm_choice == "Llama (Local)":
            st.session_state["llm_type"] = "llama"
            st.session_state["llm"] = Llama(model_path=MODEL_PATH, **LLM_CONFIG)
        elif llm_choice == "GPT (API)":
            st.session_state["llm_type"] = "openai"
            st.session_state["llm"] = OpenAIChatWrapper(model="gpt-4.1-mini")

        st.session_state["embedder"] = SentenceTransformer(EMBEDDING_MODEL)
        st.session_state["reranker"] = CrossEncoder(RERANKER_MODEL)
        st.session_state["models_loaded"] = True

# Giao diện StreamGPT
st.set_page_config(page_title="Chatbot Y tế", layout="wide")
st.title("🧠 Hệ thống Chatbot Y tế")

llm_choice = st.selectbox("Chọn LLM", ["Llama (Local)", "GPT (API)"])

if st.button("🚀 Khởi tạo mô hình"):
    init_models(llm_choice)
    st.success(f"Đã khởi tạo mô hình với: {llm_choice}")
