import streamlit as st
import time
from config import CHROMA_COLLECTION_NAME, GOOGLE_API_KEY, GOOGLE_CSE_ID
from modules.retriever.query_analyzer import analyze_query_with_llm
from modules.retriever.retriever import retrieve_data
from modules.retriever.context_builder import build_context
from modules.retriever.log_manager import save_log, save_llm_analysis_log
from modules.retriever.response_generator import generate_response_stream
from modules.llm_utils import openai_chat_completion_stream, llama_chat_completion_stream
from modules.google_search_tamanh import GoogleSearchTamAnh


google_searcher = GoogleSearchTamAnh(GOOGLE_API_KEY, GOOGLE_CSE_ID)
st.title("💬 Trò chuyện với Chatbot Y tế")

if "client" not in st.session_state:
    st.warning("Vui lòng khởi tạo mô hình và cơ sở dữ liệu từ trang chính.")
    st.stop()
client = st.session_state.client
collection = client.get_collection(name=CHROMA_COLLECTION_NAME)

def call_llm_stream(messages, model_type="llama", llm=None):
    if model_type == "openai":
        # Bạn chọn model openai hay qwen qua session_state
        yield from openai_chat_completion_stream(messages, model="gpt-4o-mini") # Hoặc model khác
    else:
        yield from llama_chat_completion_stream(llm, messages)

# Lưu lại loại model
if "llm_type" not in st.session_state:
    st.session_state["llm_type"] = "llama"  # hoặc "openai"

if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    # 1) Phân tích truy vấn
    section_ids, main_obj, disease_names, analysis_dict, raw_resp, analysis_time = \
        analyze_query_with_llm(prompt, st.session_state.llm)
    save_llm_analysis_log(
        user_input=prompt,
        llm_response=raw_resp,
        llm_analysis=analysis_dict,
        analysis_time=analysis_time
    )

    # 2) Truy xuất dữ liệu
    docs, metas = retrieve_data(
        collection=collection,
        embedder=st.session_state.embedder,
        reranker=st.session_state.reranker,
        query_disease_name=disease_names,
        query_section_id=main_obj,
        section_ids=section_ids,
        query_text=prompt,
        top_k=5
    )
    # if not docs:
        # Nếu không có dữ liệu, gọi Google
    context = google_searcher.get_context(prompt, section_ids, main_obj)
    #     if not context:
    #         context = "Không tìm thấy thông tin liên quan trên website Tâm Anh hoặc Google."
    # else:
    #     context = build_context(docs, metas)

    # 4) Sinh phản hồi và stream
    # Gọi hàm generate_response_stream
    gen, get_full_response, get_response_time, messages = generate_response_stream(prompt, context, st.session_state.llm)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        # Stream ra UI
        for chunk in gen:
            placeholder.markdown(get_full_response())  # luôn lấy full để đảm bảo hiển thị đầy đủ

    # Sau khi stream xong, lấy lại kết quả:
    full_response = get_full_response()
    response_time = get_response_time()

    # Lưu log
    save_log(
        user_message=prompt,
        bot_response=full_response,
        response_time=response_time,
        docs=docs,
        metas=metas,
        prompt=messages,
        llm_analysis=analysis_dict
    )
