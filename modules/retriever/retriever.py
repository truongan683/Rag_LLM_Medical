import logging
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pyvi.ViTokenizer import ViTokenizer
from chromadb import Client
from chromadb.config import Settings
from modules.retriever.utils import normalize_doc_name, find_disease_fuzzy

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Cache embeddings để tái sử dụng
_embedding_cache: Dict[str, np.ndarray] = {}

# Danh sách stop words (cần bổ sung)
STOP_WORDS = set([
    'a', 'là', 'các', 'cái', 'có', 'có_thể', 'chính', 'cho', 'chứ', 'chưa', 'chúng', 'chúng_tôi', 'chúng_ta',
    'chúng_mình', 'chúng_tôi', 'chúng_tớ', 'cùng', 'của', 'cuối', 'cứ', 'cũng', 'đã', 'đang', 'đây',
    'đó', 'được', 'dưới', 'em', 'gì', 'hoặc', 'hơn', 'khi', 'không', 'lại', 'lên', 'lúc', 'mà', 'mỗi',
    'một', 'mỗi_một', 'mình', 'nào', 'này', 'nên', 'nếu', 'ngay', 'người', 'nhiều', 'như', 'nhưng',
    'nữa', 'phải', 'qua', 'ra', 'rằng', 'rất', 'rồi', 'sau', 'sẽ', 'so', 'theo', 'thì', 'trên',
    'trong', 'trước', 'từ', 'từng', 'và', 'vẫn', 'vào', 'vì', 'với', 'vừa', 'đến', 'đi', 'ở', 'ơi'
])


def tokenize_vietnamese(text: str) -> List[str]:
    """
    Tách tiếng Việt thành token, loại bỏ stop words và ký tự đặc biệt.
    """
    tokens = ViTokenizer.tokenize(text).lower().split()
    clean_tokens = [re.sub(r"[^\w]+", "", tok) for tok in tokens]
    return [tok for tok in clean_tokens if tok and tok not in STOP_WORDS]

def get_metadata_values(collection) -> Tuple[List[str], List[str]]:
    """
    Trả về (danh sách tên bệnh, danh sách section_id) có trong collection.
    """
    data = collection.get(include=["metadatas"])
    metas = data.get("metadatas", [])
    names = set()
    secs = set()
    for m in metas:
        dn = m.get("doc_name")
        sid = m.get("section_id")
        if dn:
            names.add(dn)
        if sid:
            secs.add(sid)
    return list(names), list(secs)

def rerank_documents(
    query_text: str,
    docs: List[str],
    metas: List[Dict[str, Any]],
    reranker: Any,
    top_k: int
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Rerank documents using model xếp hạng và logging để debug
    """
    logger.info("Starting reranking documents")
    # Tạo list cặp (query, doc)
    pairs = [[query_text, doc] for doc in docs]
    # Dự đoán điểm xếp hạng
    scores = reranker.predict(pairs)
    logger.debug(f"Rerank scores: {scores}")
    # Sắp xếp theo score giảm dần
    order = np.argsort(scores)[::-1][:top_k]
    logger.info(f"Documents reranked to new indices: {order}")
    reranked_docs = [docs[i] for i in order]
    reranked_metas = [metas[i] for i in order]
    return reranked_docs, reranked_metas


def get_documents_in_chroma(
    collection: Any,
    filt: Optional[Dict[str, Any]],
    query_text: Optional[str],
    embedder: SentenceTransformer,
    reranker: Optional[Any],
    top_k: int
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Truy xuất tài liệu từ ChromaDB, sử dụng BM25 và semantic reranking.
    """
    logger.info(f"Querying Chroma with filter: {filt} and query_text: {query_text}")
    # Lấy docs và metadata
    res = collection.get(where=filt, include=["documents", "metadatas"])
    docs = res["documents"]
    metas = res["metadatas"]
    ids = res["ids"]
    if query_text:
        # BM25 filter thô
        logger.info("Starting BM25 retrieval")
        tokenized_corpus = [tokenize_vietnamese(doc) for doc in docs]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = tokenize_vietnamese(query_text)
        scores = bm25.get_scores(tokenized_query)
        logger.debug(f"BM25 scores: {list(zip(ids, scores))}")
        # Chọn top 3*top_k docs
        bm25_indices = np.argsort(scores)[::-1][:3 * top_k]
        bm25_selected_ids = [ids[i] for i in bm25_indices]
        bm25_docs = [docs[i] for i in bm25_indices]
        bm25_metas = [metas[i] for i in bm25_indices]
        logger.info(f"BM25 selected {len(bm25_selected_ids)} documents for semantic reranking")

        # Semantic similarity
        logger.info("Starting semantic similarity retrieval")
        query_emb = embedder.encode([query_text], normalize_embeddings=True)[0]
        logger.debug(f"Query embedding sample: {query_emb[:5]}...")
        doc_embeddings = []
        for doc_id, doc in zip(bm25_selected_ids, bm25_docs):
            if doc_id in _embedding_cache:
                emb = _embedding_cache[doc_id]
                logger.debug(f"Using cached embedding for doc {doc_id}")
            else:
                emb = embedder.encode([doc], normalize_embeddings=True)[0]
                _embedding_cache[doc_id] = emb
                logger.debug(f"Computed and cached embedding for doc {doc_id}")
            doc_embeddings.append(emb)
        cos_scores = cosine_similarity([query_emb], doc_embeddings)[0]
        logger.debug(f"Semantic cosine scores: {list(zip(bm25_selected_ids, cos_scores))}")
        sem_indices = np.argsort(cos_scores)[::-1][:3 * top_k]
        sem_docs = [bm25_docs[i] for i in sem_indices]
        sem_metas = [bm25_metas[i] for i in sem_indices]
        sem_ids = [bm25_selected_ids[i] for i in sem_indices]
        logger.info(f"Semantic selected {len(sem_ids)} documents before final rerank")

        # Nếu có reranker thì rerank
        if reranker:
            sem_docs, sem_metas = rerank_documents(query_text, sem_docs, sem_metas, reranker, top_k)
        else:
            sem_docs = sem_docs[:top_k]
            sem_metas = sem_metas[:top_k]
        return sem_docs, sem_metas
    else:
        # Không có query_text, chỉ trả về toàn bộ tài liệu (có thể giới hạn)
        logger.info("No query text provided, returning up to top_k documents without reranking")
        return docs[:top_k], metas[:top_k]


def retrieve_data(
    collection,
    embedder,
    reranker,
    query_disease_name: Optional[object] = None,
    query_section_id: Optional[str] = None,
    section_ids: Optional[List[str]] = None,
    query_text: Optional[str] = None,
    top_k: int = 5
) -> Tuple[List[str], List[dict]]:
    """
    Truy xuất dữ liệu từ ChromaDB dựa trên tên bệnh và ID phần.

    Args:
        collection: ChromaDB collection đã được khởi tạo.
        embedder: Model embedding.
        reranker: Model reranker.
        query_disease_name (str hoặc list): Tên bệnh để tìm kiếm.
        query_section_id (str): ID phần chính để xử lý trường hợp xếp hạng vòng 2.
        section_ids (list): Danh sách ID phần để lọc.
        query_text (str): Nội dung truy vấn để BM25/Semantic.
        top_k (int): Số lượng kết quả trả về.

    Returns:
        Tuple[List[str], List[dict]]: Danh sách document và metadatas tương ứng.
    """
    # 0. Lấy thông tin metadata hiện có
    disease_names, section_types = get_metadata_values(collection)

    # 1) Nếu có tên bệnh -> lọc theo bệnh
    if query_disease_name:
        all_docs: List[str] = []
        all_metas: List[dict] = []
        if isinstance(query_disease_name, str):
            query_disease_name = [query_disease_name]
        for qd in query_disease_name:
            norm = normalize_doc_name(qd)
            match = find_disease_fuzzy(norm, disease_names)
            if not match:
                continue
            # Lọc thêm theo section_ids nếu có
            if section_ids:
                filters = {
                    "$and": [
                        {"doc_name": match},
                        {"section_id": {"$in": section_ids}}
                    ]
                }
            else:
                filters = {"doc_name": match}
            res = collection.get(where=filters, include=["documents", "metadatas"])
            all_docs.extend(res.get("documents", []))
            all_metas.extend(res.get("metadatas", []))
        return all_docs, all_metas

    # 2) Nếu không có tên bệnh nhưng có section_ids
    if section_ids:
        # 2a) Trường hợp phân nhánh vòng 2: nhiều section và có section chính
        if len(section_ids) > 1 and query_section_id:
            # Lọc bỏ section chính để tìm các section hỗ trợ
            filtered = [sid for sid in section_ids if sid != query_section_id]
            # Bước 1: lấy candidate
            cand_docs, cand_metas = get_documents_in_chroma(
                collection,
                {"section_id": {"$in": filtered}},
                query_text, embedder, reranker,
                top_k * 3
            )
            # Bước 2: rerank top_k
            final_docs, final_metas = rerank_documents(
                query_text, cand_docs, cand_metas, reranker, top_k
            )
            # Bước 3: lấy tên bệnh mới từ metadata
            new_names = [m.get("doc_name") for m in final_metas if m.get("doc_name")]
            if new_names:
                # Gọi lại chính hàm này với tên bệnh mới và chỉ section chính
                return retrieve_data(
                    collection,
                    embedder,
                    reranker,
                    query_disease_name=new_names,
                    query_section_id=query_section_id,
                    section_ids=[query_section_id],
                    query_text=query_text,
                    top_k=top_k
                )
            return [], []

        # 2b) Trường hợp chỉ lọc theo section_ids duy nhất
        filters = {"section_id": {"$in": section_ids}}
        return get_documents_in_chroma(
            collection,
            filters,
            query_text, embedder, reranker,
            top_k
        )

    # 3) Nếu không có tham số nào phù hợp -> trả về rỗng
    return [], []