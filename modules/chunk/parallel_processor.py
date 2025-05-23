import os
from concurrent.futures import ThreadPoolExecutor
import logging
from .document_processor import process_document
from config import FOLDER_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def process_documents_parallel(folder_path: str = FOLDER_PATH,
                               tokenizer=None,
                               model=None) -> list:
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".txt")]
    if not files:
        logging.error("Không tìm thấy file .txt nào!")
        return []
    documents, counter = [], 0
    total = len(files)
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_document, path, idx, total, counter + idx, tokenizer, model)
                   for idx, path in enumerate(files)]
        for future in futures:
            try:
                data, new_cnt = future.result()
                documents.extend(data)
                counter = max(counter, new_cnt)
            except Exception as e:
                logging.error(f"Error: {e}")
    logging.info(f"Tạo được {len(documents)} chunks từ {total} files")
    return documents