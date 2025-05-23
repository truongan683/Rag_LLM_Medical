import logging
from config import SECTION_CATEGORIES

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def determine_section_id_from_title(title):
    title_lower = title.lower().strip()
    logging.debug(f"Đang kiểm tra tiêu đề: '{title_lower}'")
    if "triệu chứng" in title_lower or "dấu hiệu" in title_lower:
        return "triệu chứng"
    elif "phòng ngừa" in title_lower or "phòng tránh" in title_lower:
        return "phòng ngừa"
    for category in SECTION_CATEGORIES[:-1]:
        if category in title_lower:
            return category
    return "khác"