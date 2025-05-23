"""
Ghi lại log của người dùng và phân tích LLM vào file JSON.
"""
import json
import os
from datetime import datetime
from config import LOG_FILE, LLM_ANALYSIS_LOG_FILE

# Đảm bảo thư mục chứa file log tồn tại
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
os.makedirs(os.path.dirname(LLM_ANALYSIS_LOG_FILE), exist_ok=True)

def save_log(user_message: str,
             bot_response: str,
             response_time: float,
             docs: list,
             metas: list,
             prompt: str,
             llm_analysis: dict = None) -> None:
    """Lưu log chính khi chat."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "user_message": user_message,
        "bot_response": bot_response,
        "response_time_seconds": round(response_time, 2),
        "chromadb_data": {"documents": docs or [], "metadatas": metas or []},
        "prompt": prompt,
        "llm_analysis": llm_analysis
    }
    logs = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            logs = json.load(f)
    logs.append(entry)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=4)

def save_llm_analysis_log(user_input: str,
                          llm_response: str,
                          llm_analysis: dict,
                          analysis_time: float) -> None:
    """Lưu log phân tích của LLM."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "user_input": user_input,
        "llm_response": llm_response.strip(),
        "llm_analysis": llm_analysis,
        "analysis_time_seconds": round(analysis_time, 2)
    }
    analyses = []
    if os.path.exists(LLM_ANALYSIS_LOG_FILE):
        with open(LLM_ANALYSIS_LOG_FILE, "r", encoding="utf-8") as f:
            analyses = json.load(f)
    analyses.append(entry)
    with open(LLM_ANALYSIS_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(analyses, f, ensure_ascii=False, indent=4)