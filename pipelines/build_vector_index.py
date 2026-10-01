"""
Simple vector index framework for initial testing and development. This is a basic implementation that allows you to create a vector index from a set of documents and perform similarity searches.
"""

import chromadb
from sentence_transformers import SentenceTransformer
from analytics.load_data import load_fault_codes, load_instructions, load_fault_logs
from app.config import settings

EMBED_MODEL = settings.EMBEDDING_MODEL
CHROMA_PATH = settings.VECTOR_STORE_DIR
COLLECTION  = settings.VECTOR_COLLECTION

_model = None



def embed(texts, is_query=False):
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    prefix = "query: " if is_query else "passage: "
    return _model.encode([prefix + t for t in texts], normalize_embeddings=True).tolist()





def build_index():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:                                   
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    _, code_records = load_fault_codes()
    records = load_instructions() + code_records + load_fault_logs()
    if not records:
        raise SystemExit("No data found — check the paths in the Config section.")

    ids   = [r[0] for r in records]
    docs  = [r[1] for r in records]
    metas = [r[2] for r in records]
    col.add(ids=ids, documents=docs, metadatas=metas, embeddings=embed(docs))
    print(f"Indexed {len(ids)} chunks "
          f"({sum(m['source_type']=='instruction' for m in metas)} instr, "
          f"{sum(m['source_type']=='code' for m in metas)} codes, "
          f"{sum(m['source_type']=='log' for m in metas)} logs).")

