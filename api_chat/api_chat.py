from modules.retriever.query_analyzer import analyze_query_with_llm
from modules.retriever.retriever import retrieve_data
from modules.retriever.context_builder import build_context
from modules.retriever.response_generator import generate_response_stream
from modules.retriever.log_manager import save_log, save_llm_analysis_log
from config import GOOGLE_API_KEY, GOOGLE_CSE_ID
from modules.google_search_tamanh import GoogleSearchTamAnh

google_searcher = GoogleSearchTamAnh(GOOGLE_API_KEY, GOOGLE_CSE_ID)
def process_prompt(prompt: str, llm, collection, embedder, reranker, model_type="openai"):
    """
    Hàm xử lý prompt, giống logic trong Streamlit, nhưng dùng cho API FastAPI.
    Trả về full_response để trả lời qua endpoint.
    """

    # 1. Phân tích truy vấn bằng LLM
    section_ids, main_obj, disease_names, analysis_dict, raw_resp, analysis_time = \
        analyze_query_with_llm(prompt, llm["client"])  # truyền vào đúng llm client (OpenAI hoặc Llama)

    save_llm_analysis_log(
        user_input=prompt,
        llm_response=raw_resp,
        llm_analysis=analysis_dict,
        analysis_time=analysis_time
    )

    # 2. Truy xuất dữ liệu từ ChromaDB
    docs, metas = retrieve_data(
        collection=collection,
        embedder=embedder,
        reranker=reranker,
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

    # 4. Sinh phản hồi và stream
    gen, get_full_response, get_response_time, messages = generate_response_stream(
        prompt=prompt,
        context=context,
        llm=llm["client"]
    )

    # Stream toàn bộ để lấy kết quả hoàn chỉnh
    for _ in gen:
        pass

    full_response = get_full_response()
    response_time = get_response_time()

    # 5. Lưu log
    save_log(
        user_message=prompt,
        bot_response=full_response,
        response_time=response_time,
        docs=docs,
        metas=metas,
        prompt=messages,
        llm_analysis=analysis_dict
    )

    return full_response
