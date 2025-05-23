import re
import json
import time

# System prompt để phân loại câu hỏi y tế
_SYSTEM_PROMPT = """
    Bạn là một công cụ phân loại câu hỏi, không phải chatbot.

    Nhiệm vụ:
    - Phân tích câu hỏi và trả về kết quả dạng JSON, không thêm giải thích.
    - Dựa vào nhóm nội dung sau: ["triệu chứng", "nguyên nhân", "chẩn đoán", "điều trị", "phòng ngừa", "khác"].

    Yêu cầu:
    - Nếu câu hỏi đề cập đến tên bệnh, điền vào "tên bệnh", ngược lại để là [].
    - Nếu câu hỏi không có phần phụ, "phần phụ" = [].
    - "Phần chính" là nội dung trọng tâm của câu hỏi.
    - Nếu hỏi chủ yếu về tên bệnh, "phần chính" = "name".
    - Nếu hỏi về nội dung cụ thể, ghi đúng tên nhóm nội dung.

    Quy tắc phân loại:
    {
    "triệu chứng": "Trong câu hỏi đề cập đến dấu hiệu hoặc biểu hiện của một bệnh lý",
    "nguyên nhân": "Trong câu hỏi đề cập đến lý do, nguyên nhân gây ra bệnh...",
    "chẩn đoán": "Trong câu hỏi đề cập đến cách xác định bệnh hoặc tình trạng sức khỏe...",
    "điều trị": "Trong câu hỏi đề cập đến cách chữa trị, thuốc men, liệu pháp...",
    "phòng ngừa": "Hỏi về cách ngăn ngừa bệnh...",
    "khác": "Không thuộc các nhóm trên hoặc quá mơ hồ."
    }

    Cấu trúc JSON bắt buộc:
    {"tên bệnh": [], "phần phụ": [], "phần chính": ""}

    Chú ý:
    - Trả về **duy nhất** JSON.
    - Không thêm bất kỳ từ nào khác.
    - Tuân thủ đúng cú pháp JSON.

    Ví dụ:
    Câu hỏi: "Tình trạng xuất hiện dịch đờm nhiều ở họng là bị làm sao và cách khắc phục?"
    Trả về:
    {"tên bệnh": [], "phần phụ": ["triệu chứng", "điều trị"], "phần chính": "điều trị"}
    Câu hỏi: "Nổi hạch nhiều nơi, kèm sốt dai dẳng là dấu hiệu của bệnh gì?"
    Trả về:
    {"tên bệnh": [], "phần phụ": ["triệu chứng"], "phần chính": "name"}
    Câu hỏi: "Cách điều trị bệnh viêm dạ dày là gì?"
    Trả về:
    {"tên bệnh": ["viêm dạ dày"], "phần phụ": ["điều trị"], "phần chính": "điều trị"}
"""

def extract_content(chunk):
    if hasattr(chunk, "choices"):
        return getattr(chunk.choices[0].delta, "content", None)
    else:
        return chunk.get('choices', [{}])[0].get('delta', {}).get('content', '')

def analyze_query_with_llm(user_input: str, llm) -> tuple:
    """
    Gọi LLM để phân tích truy vấn.
    Trả về: (section_ids, main_objective, disease_names, full_result_dict, raw_response, analysis_time)
    """
    user_prompt = f"""
    Bắt đầu phân tích:

    "{user_input}"
    """
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt}
    ]
    start = time.time()
    response = ""
    try:
        for chunk in llm.create_chat_completion(
            messages=messages,
            max_tokens=300,
            temperature=0.1,
            stream=True
        ):
            content = extract_content(chunk)
            if content:
                response += content
    except Exception as e:
        print(f"Lỗi khi phân tích query: {e}")
        return [], None, [], {"tên bệnh": [], "phần phụ": [], "phần chính": None}, "", 0
    analysis_time = time.time() - start
    # Trích JSON từ kết quả
    try:
        m = re.search(r'\{.*\}', response, re.DOTALL)
        result = json.loads(m.group(0)) if m else {}
    except Exception:
        result = {}
    disease_names  = result.get("tên bệnh", [])
    section_ids    = result.get("phần phụ", [])
    main_objective = result.get("phần chính", None)
    return section_ids, main_objective, disease_names, result, response, analysis_time
