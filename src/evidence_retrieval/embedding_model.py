from typing import List, Union
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from src.utils.logging_utils import get_logger

logger = get_logger("EmbeddingModel")

class EmbeddingModel:
    """Wrapper around SentenceTransformer for generating normalized semantic embeddings."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = None):
        self.model_name = model_name
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Loading embedding model '{model_name}' on device: {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)
        if hasattr(self.model, "get_embedding_dimension"):
            self.embedding_dim = self.model.get_embedding_dimension()
        else:
            self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        normalize_embeddings: bool = True
    ) -> np.ndarray:
        """Encodes texts into numpy embedding vectors.
        Normalizes embeddings by default so dot product equals cosine similarity.
        """
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize_embeddings
        )
        return embeddings.astype(np.float32)
