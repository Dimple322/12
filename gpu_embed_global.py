# gpu_embed_global.py
import torch
from sentence_transformers import SentenceTransformer
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


class GpuMiniLMEmbedding(SentenceTransformerEmbeddingFunction):
    """
    Drop-in replacement that forces GPU/CPU the same way Chroma expects.
    We inherit from SentenceTransformerEmbeddingFunction so the *type*
    stored in the collection meta is always sentence_transformer
    -> no conflict on re-open.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        super().__init__(model_name=model_name, device=device)


# global singleton – import & reuse everywhere
gpu_embedding = GpuMiniLMEmbedding()