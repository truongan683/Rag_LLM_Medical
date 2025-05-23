
# modules/llm_utils.py
import openai
from config import OPENAI_API_KEY  # Nếu cần truyền key rõ ràng

class OpenAIChatWrapper:
    def __init__(self, model="gpt-4o"):
        # Nếu muốn lấy key từ biến môi trường, bỏ api_key, nếu không thì truyền luôn api_key
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.model = model

    def create_chat_completion(self, messages, max_tokens=500, temperature=0.7, stream=True):
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream
        )

def openai_chat_completion_stream(messages, model="gpt-4o", max_tokens=500, temperature=0.7):
    # Dùng key nếu cần, hoặc mặc định lấy từ env
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True
    )
    for chunk in response:
        # chunk.choices[0].delta.content là content mới sinh ra ở mỗi chunk (object, không phải dict)
        content = getattr(chunk.choices[0].delta, "content", None)
        if content:
            yield content

def llama_chat_completion_stream(llm, messages, max_tokens=500, temperature=0.7):
    for chunk in llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True
    ):
        # Llama vẫn trả về dict cũ
        delta = chunk.get("choices", [{}])[0].get("delta", {})
        yield delta.get('content', '')
