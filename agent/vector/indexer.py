# agent/vector/indexer.py (локальная загрузка моделей + безопасный fallback)
import os
import torch
from pathlib import Path
from typing import List, Dict

import chromadb
from sentence_transformers import SentenceTransformer

# Путь к локальной модели (в репозитории)
MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "sentence"

# Попытка загрузить модель локально; если не найдено — выдаём понятный warning и отключаем энкодер
ENCODER = None
try:
    if MODEL_DIR.exists() and any(MODEL_DIR.iterdir()):
        # Используем локальную папку модели
        ENCODER = SentenceTransformer(str(MODEL_DIR))
        print(f"[INDEXER] Используется локальная модель SentenceTransformer из: {MODEL_DIR}")
    else:
        # Попытка загрузить по имени (но это потребует интернета)
        print("[INDEXER] Локальная модель не найдена, пытаемся загрузить по имени (нужно интернет).")
        ENCODER = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
except Exception as e:
    print(f"[INDEXER WARNING] Не удалось инициализировать SentenceTransformer: {e}")
    ENCODER = None

def index_documents(texts: List[str], metas: List[Dict[str, str]]):
    if not texts:
        print("⚠️ Нет текстов для индексации")
        return

    db_path = Path(__file__).resolve().parent.parent.parent / "generated" / "chroma_db"
    db_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(db_path))
    coll = client.get_or_create_collection("docs", metadata={"hnsw:space": "cosine"})

    if ENCODER is None:
        print("⚠️ ENCODER не инициализирован, индексация пропущена.")
        return

    print(f"🧠 Кодируем {len(texts)} фрагментов...")
    embeddings = ENCODER.encode(texts, normalize_embeddings=True, show_progress_bar=True).tolist()

    ids = [f"id_{i}" for i in range(len(texts))]
    coll.add(documents=texts, embeddings=embeddings, metadatas=metas, ids=ids)
    print(f"✅ Индексировано {len(texts)} фрагментов (локально)")