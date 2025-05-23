import time

def extract_content(chunk):
    if hasattr(chunk, "choices"):
        return getattr(chunk.choices[0].delta, "content", None)
    else:
        return chunk.get('choices', [{}])[0].get('delta', {}).get('content', '')

def generate_response_stream(prompt: str, context: str, llm, max_tokens=500, temperature=0.7):
    """
    Stream từng chunk từ LLM khi trả lời, đồng thời trả về full response, thời gian và messages để log.
    Trả về: generator (chunk), full_response, response_time, messages
    """
    system_prompt = """
        Bạn là một chuyên gia y tế.
        - Chỉ dựa vào nội dung được cung cấp (context) để trả lời.
        - Không tự bịa thêm thông tin không có trong context.
        - Nếu thông tin không đủ để trả lời, hãy trả lời lịch sự rằng không đủ dữ liệu.
        - Nếu thông tin (context) không khớp với câu hỏi (ví dụ câu hỏi về bệnh "đau mắt đỏ" nhưng kết quả chỉ trả về "đau mắt"), hãy trả lời rằng không đủ dữ liệu.
        - Nếu câu hỏi không liên quan đến y tế, hãy trả lời rằng không phải lĩnh vực của bạn.
        - Chỉ trả lời bằng tiếng Việt.
    """
    user_prompt = f"""
    Dưới đây là một số thông tin y tế (context):

    {context}

    Câu hỏi:
    {prompt}
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    start = time.time()
    full_response = ""

    def chunk_generator():
        nonlocal full_response
        try:
            for chunk in llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True
            ):
                content = extract_content(chunk)
                if content:
                    full_response += content
                    yield content
        except Exception as e:
            print(f"Lỗi khi tạo phản hồi: {e}")

    response_time = None
    def run_and_get_all():
        nonlocal response_time
        for chunk in chunk_generator():
            yield chunk
        response_time = time.time() - start

    return run_and_get_all(), lambda: full_response, lambda: response_time, messages
