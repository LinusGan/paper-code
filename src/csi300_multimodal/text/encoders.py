from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from csi300_multimodal.text.aggregation import embedding_columns

POSITIVE_KEYWORDS = ("rise", "gain", "up", "bull", "profit", "improve", "optimistic", "growth")
NEGATIVE_KEYWORDS = ("fall", "drop", "down", "bear", "loss", "risk", "decline", "warning")


@dataclass
class EncodedTextBatch:
    probabilities: np.ndarray
    embeddings: np.ndarray


class BaseTextEncoder:
    def __init__(self, embedding_dim: int) -> None:
        self.embedding_dim = embedding_dim

    def encode_batch(self, texts: Iterable[str]) -> EncodedTextBatch:
        raise NotImplementedError


class DummyTextEncoder(BaseTextEncoder):
    def encode_batch(self, texts: Iterable[str]) -> EncodedTextBatch:
        values = list(texts)
        if not values:
            return EncodedTextBatch(
                probabilities=np.zeros((0, 3), dtype=float),
                embeddings=np.zeros((0, self.embedding_dim), dtype=float),
            )
        probs: list[np.ndarray] = []
        embeds: list[np.ndarray] = []
        for text in values:
            lowered = (text or "").lower()
            pos_hits = sum(keyword in lowered for keyword in POSITIVE_KEYWORDS)
            neg_hits = sum(keyword in lowered for keyword in NEGATIVE_KEYWORDS)
            if pos_hits > neg_hits:
                prob = np.array([0.7, 0.1, 0.2], dtype=float)
            elif neg_hits > pos_hits:
                prob = np.array([0.1, 0.7, 0.2], dtype=float)
            else:
                prob = np.array([0.1, 0.1, 0.8], dtype=float)
            digest = hashlib.sha256((text or "").encode("utf-8")).digest()
            raw = np.frombuffer(digest, dtype=np.uint8).astype(float)
            tiled = np.resize(raw, self.embedding_dim)
            embed = (tiled / 255.0) * 2.0 - 1.0
            probs.append(prob)
            embeds.append(embed.astype(float))
        return EncodedTextBatch(probabilities=np.vstack(probs), embeddings=np.vstack(embeds))


class TransformersTextEncoder(BaseTextEncoder):
    def __init__(
        self,
        embedding_dim: int,
        device: str,
        max_length: int,
        batch_size: int,
        model_name_or_path: str,
        sentiment_model_name_or_path: str | None = None,
        embedding_model_name_or_path: str | None = None,
    ) -> None:
        super().__init__(embedding_dim=embedding_dim)
        try:
            import torch
        except Exception as exc:
            raise ImportError("torch is required for backend='transformers'.") from exc
        try:
            from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ImportError("transformers is required for backend='transformers'.") from exc

        sentiment_path = sentiment_model_name_or_path or model_name_or_path
        embedding_path = embedding_model_name_or_path or model_name_or_path
        if not sentiment_path or not embedding_path:
            raise ValueError("A model path is required for transformers backend.")

        self.torch = torch
        self.device = torch.device(device)
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path or sentiment_path)
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_path).to(self.device)
        self.embedding_model = AutoModel.from_pretrained(embedding_path).to(self.device)
        self.sentiment_model.eval()
        self.embedding_model.eval()

    def encode_batch(self, texts: Iterable[str]) -> EncodedTextBatch:
        values = list(texts)
        if not values:
            return EncodedTextBatch(
                probabilities=np.zeros((0, 3), dtype=float),
                embeddings=np.zeros((0, self.embedding_dim), dtype=float),
            )
        torch = self.torch
        tokenized = self.tokenizer(
            values,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tokenized = {key: value.to(self.device) for key, value in tokenized.items()}
        with torch.no_grad():
            logits = self.sentiment_model(**tokenized).logits
            probabilities = torch.softmax(logits, dim=-1)
            hidden = self.embedding_model(**tokenized).last_hidden_state[:, 0, :]
        prob_np = probabilities.detach().cpu().numpy()
        emb_np = hidden.detach().cpu().numpy()
        if emb_np.shape[1] != self.embedding_dim:
            if emb_np.shape[1] > self.embedding_dim:
                emb_np = emb_np[:, : self.embedding_dim]
            else:
                padding = np.zeros((emb_np.shape[0], self.embedding_dim - emb_np.shape[1]), dtype=emb_np.dtype)
                emb_np = np.concatenate([emb_np, padding], axis=1)
        return EncodedTextBatch(probabilities=prob_np, embeddings=emb_np)


def build_text_encoder(config: dict) -> BaseTextEncoder:
    backend = config["text"]["backend"]
    if backend == "dummy":
        return DummyTextEncoder(embedding_dim=int(config["text"]["embedding_dim"]))
    if backend == "transformers":
        return TransformersTextEncoder(
            embedding_dim=int(config["text"]["embedding_dim"]),
            device=str(config["text"]["device"]),
            max_length=int(config["text"]["max_length"]),
            batch_size=int(config["text"]["batch_size"]),
            model_name_or_path=str(config["text"].get("model_name_or_path", "")),
            sentiment_model_name_or_path=config["text"].get("sentiment_model_name_or_path") or None,
            embedding_model_name_or_path=config["text"].get("embedding_model_name_or_path") or None,
        )
    raise ValueError(f"Unsupported text backend: {backend}")


def encode_news_frame(news_frame: pd.DataFrame, encoder: BaseTextEncoder, text_column: str) -> pd.DataFrame:
    frame = news_frame.copy()
    columns = embedding_columns(encoder.embedding_dim)
    if frame.empty:
        frame["pos"] = pd.Series(dtype=float)
        frame["neg"] = pd.Series(dtype=float)
        frame["neu"] = pd.Series(dtype=float)
        for name in columns:
            frame[name] = pd.Series(dtype=float)
        return frame
    texts = frame[text_column].fillna("").tolist()
    encoded = encoder.encode_batch(texts)
    frame["pos"] = encoded.probabilities[:, 0]
    frame["neg"] = encoded.probabilities[:, 1]
    frame["neu"] = encoded.probabilities[:, 2]
    for index, name in enumerate(columns):
        frame[name] = encoded.embeddings[:, index]
    return frame
