import hashlib
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def generate_unique_id(doc_name, section_id, chunk_index, counter):
    raw_id = f"{doc_name}_{section_id}_{chunk_index}_{counter}_{int(time.time())}"
    return hashlib.md5(raw_id.encode()).hexdigest()