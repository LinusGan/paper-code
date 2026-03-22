from __future__ import annotations

from csi300_multimodal.config import resolve_path
from csi300_multimodal.text.encoders import build_text_encoder, encode_news_frame
from csi300_multimodal.utils.io import read_table, write_table


def run_encode_text(config: dict) -> None:
    aligned_news = read_table(resolve_path(config, "aligned_news_path"))
    encoder = build_text_encoder(config)
    text_column = str(config["text"]["text_column"])
    if text_column not in aligned_news.columns:
        text_column = "content" if "content" in aligned_news.columns else aligned_news.columns[0]
    encoded = encode_news_frame(aligned_news, encoder, text_column=text_column)
    write_table(encoded, resolve_path(config, "encoded_news_path"))
