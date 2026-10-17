from typing import Tuple, Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.utils.logging_utils import get_logger

logger = get_logger("ModelLoader")

class ModelLoader:
    """Manages cached loading and device allocation for transformer verification models."""

    _cached_models: Dict[str, Tuple[Any, Any]] = {}

    @classmethod
    def load_nli_model(
        cls,
        model_name: str = "cross-encoder/nli-distilroberta-base",
        device: Optional[str] = None
    ) -> Tuple[Any, Any, str]:
        """Loads and caches a sequence classification model and tokenizer for NLI."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        cache_key = f"{model_name}_{device}"
        if cache_key in cls._cached_models:
            tokenizer, model = cls._cached_models[cache_key]
            return tokenizer, model, device

        logger.info(f"Loading transformer model '{model_name}' on device: {device}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.to(device)
        model.eval()

        cls._cached_models[cache_key] = (tokenizer, model)
        return tokenizer, model, device
