import torch
from transformers import AutoTokenizer, AutoModel
from torch.amp import autocast
import logging
from config import EMBEDDING_MODEL, DEVICE

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

tokenizer_embedding = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
model_embedding = AutoModel.from_pretrained(EMBEDDING_MODEL).to(DEVICE)

def embed_texts(texts, batch_size=32):
    embeddings = []
    model_embedding.eval()
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer_embedding(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to(DEVICE)
        with torch.no_grad():
            with autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model_embedding(**inputs)
        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.extend(batch_embeddings)
    return embeddings

def compute_embedding(text, tokenizer=tokenizer_embedding, model=model_embedding):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()