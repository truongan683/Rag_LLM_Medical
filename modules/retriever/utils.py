"""
Các hàm hỗ trợ xử lý tên và tìm kiếm fuzzy.
"""
from unidecode import unidecode
from rapidfuzz import fuzz


def normalize_vietnamese(text: str) -> str:
    """Loại bỏ dấu và chuyển thành lowercase."""
    return unidecode(text).lower()


def normalize_doc_name(text: str) -> str:
    """Chuẩn hóa tên tài liệu: không dấu, lowercase, thay khoảng trắng thành '-'"""
    return unidecode(text).lower().replace(" ", "-")


def find_disease_fuzzy(user_input: str, disease_names: list, threshold: int = 90) -> str:
    """Tìm tên bệnh gần đúng nhất bằng fuzzy matching."""
    inp = normalize_vietnamese(user_input)
    best, score = None, 0
    for name in disease_names:
        sc = fuzz.ratio(inp, normalize_vietnamese(name))
        if sc >= threshold and sc > score:
            best, score = name, sc
    return best