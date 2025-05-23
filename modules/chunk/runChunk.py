import tempfile
import os
import logging
from .document_processor import process_document
from .embeddings import tokenizer_embedding, model_embedding
from .chromadb_manager import save_to_chromadb

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def process_and_store_documents(uploaded_file, client):
    """
    Xử lý một file được upload (UploadedFile) và lưu kết quả vào ChromaDB thông qua client đã có.
    """
    # Lưu file tạm với tên gốc trong thư mục tạm
    temp_dir = tempfile.gettempdir()
    tmp_path = os.path.join(temp_dir, uploaded_file.name)

    with open(tmp_path, 'wb') as f:
        f.write(uploaded_file.getvalue())

    # Xử lý và tạo chunks (một file duy nhất => index=0, total=1)
    chunks, _ = process_document(tmp_path, 0, 1, 0, tokenizer_embedding, model_embedding)

    # Lấy embeddings đã tính sẵn từ document_processor
    embs = [c['embedding'].flatten().tolist() if hasattr(c['embedding'], 'flatten') else c['embedding'].tolist() for c in chunks]

    # Lưu vào ChromaDB
    save_to_chromadb(chunks, embs, client)

    # Xoá file tạm
    os.remove(tmp_path)

# Giữ lại block test khi chạy script độc lập
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained('')
        model = AutoModel.from_pretrained('')
        import chromadb
        client = chromadb.PersistentClient(path='')
        process_and_store_documents(sys.argv[1], client)