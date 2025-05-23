from fastapi import FastAPI
from pydantic import BaseModel
from api_chat.api_chat import process_prompt
from api_chat.bootstrap import load_resources
import logging

# Khởi tạo FastAPI app
app = FastAPI(
    title="Chatbot Y tế API",
    description="API nhận prompt và trả lời bằng LLM",
    version="1.0.0"
)

# Biến toàn cục chứa các mô hình và tài nguyên đã load
resources = {}

# Khởi tạo tài nguyên khi app startup
@app.on_event("startup")
def startup_event():
    logging.info("Đang khởi tạo mô hình và tài nguyên...")
    llm, collection, embedder, reranker = load_resources()
    resources["llm"] = llm
    resources["collection"] = collection
    resources["embedder"] = embedder
    resources["reranker"] = reranker
    logging.info("Khởi tạo tài nguyên hoàn tất.")

# Định nghĩa request body schema
class ChatRequest(BaseModel):
    prompt: str

# Định nghĩa endpoint chat
@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    response = process_prompt(
        prompt=request.prompt,
        llm=resources["llm"],
        collection=resources["collection"],
        embedder=resources["embedder"],
        reranker=resources["reranker"],
        model_type=resources["llm"]["type"]
    )
    return {"response": response}

# (Tùy chọn) Thêm endpoint kiểm tra sức khỏe hệ thống
@app.get("/health")
def health_check():
    return {"status": "ok"}
