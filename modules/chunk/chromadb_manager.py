import chromadb
import logging
from config import CHROMA_DB_PATH, CHROMA_COLLECTION_NAME, CHROMA_METADATA

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def save_to_chromadb(processed_docs, embeddings, client=None):
    """
    Lưu danh sách chunks và embeddings vào ChromaDB.
    Nếu client đã có, sử dụng, ngược lại tạo mới từ CHROMA_DB_PATH.

    Args:
        processed_docs (List[dict]): Danh sách dict chứa 'chunk_id', 'text', 'doc_name', 'section_id'.
        embeddings (List[List[float]]): Danh sách vector tương ứng.
        client (chromadb.PersistentClient, optional): Instance client để tái sử dụng.
    """
    if client is None:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata=CHROMA_METADATA
    )

    existing_ids = set(collection.get().get('ids', []))
    data_to_add = []
    for doc, emb in zip(processed_docs, embeddings):
        cid = doc['chunk_id']
        if cid in existing_ids:
            continue
        data_to_add.append((cid, doc['text'], emb, {'doc_name': doc['doc_name'], 'section_id': doc['section_id']}))

    if not data_to_add:
        logging.warning("Không có chunks mới cần thêm vào ChromaDB!")
        return

    batch_size = 1000
    total = len(data_to_add)
    for i in range(0, total, batch_size):
        batch = data_to_add[i:i + batch_size]
        ids, docs, embs, metas = zip(*batch)
        collection.add(
            ids=list(ids),
            documents=list(docs),
            embeddings=list(embs),
            metadatas=list(metas)
        )
        logging.info(f"Đã thêm {len(batch)} chunks vào ChromaDB (batch {i // batch_size + 1})")
    logging.info(f"Tổng cộng đã thêm {total} chunks mới vào ChromaDB!")