"""
Xây dựng ngữ cảnh (context) cho LLM dựa trên các đoạn văn bản và metadata được retrieve.
"""
from typing import List

def build_context(docs: List[str], metas: List[dict]) -> str:
    """
    Nếu không có tài liệu hoặc metadata, trả về thông báo không tìm thấy.
    Ngược lại, gom từng đoạn với tên tài liệu và section ID.
    """
    if not docs or not metas:
        return "Không tìm thấy thông tin liên quan trong cơ sở dữ liệu."
    context_lines = []
    for doc, meta in zip(docs, metas):
        name = meta.get('doc_name', 'Unknown')
        section = meta.get('section_id', 'Unknown')
        snippet = doc[:200].replace("\n", " ")
        context_lines.append(f"- {name} (Phần: {section}): {snippet}...")
    return "\n".join(context_lines)