import os
import re
import logging
from .text_splitter import split_text
from .id_generator import generate_unique_id
from .section_classifier import determine_section_id_from_title
from .embeddings import compute_embedding
from config import SECTION_CATEGORIES

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def process_document(file_path: str,
                     index: int,
                     total_files: int,
                     counter: int,
                     tokenizer=None,
                     model=None) -> tuple[list, int]:
    logging.info(f"Đang xử lý file {index + 1}/{total_files}: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Tách theo heading h2/h3
    doc_name = os.path.basename(file_path).replace(".txt", "")
    sections = re.split(r'(?=\nh[23]\s)', text)
    chunk_data = []
    current_section = "khác"
    for sec in sections:
        if not sec.strip():
            continue
        m2 = re.match(r'h2\s+(.+?)\n', sec)
        m3 = re.match(r'h3\s+(.+?)\n', sec)
        if m2:
            current_section = determine_section_id_from_title(m2.group(1).strip())
            content = sec[m2.end():].strip()
        elif m3:
            content = sec[m3.end():].strip()
        else:
            content = sec.strip()
        chunks = split_text(content)
        for i, txt in enumerate(chunks):
            chunk_id = generate_unique_id(doc_name, current_section, i, counter)
            counter += 1
            embedding = compute_embedding(txt, tokenizer, model)
            chunk_data.append({
                "chunk_id": chunk_id,
                "text": txt,
                "doc_name": doc_name,
                "section_id": current_section,
                "embedding": embedding
            })
    return chunk_data, counter