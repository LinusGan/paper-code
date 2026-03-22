from __future__ import annotations

import argparse

from csi300_multimodal.config import load_config
from csi300_multimodal.pipelines.build_dataset import run_build_dataset
from csi300_multimodal.pipelines.crawl_news import run_crawl_news
from csi300_multimodal.pipelines.encode_text import run_encode_text
from csi300_multimodal.pipelines.evaluate import run_evaluate
from csi300_multimodal.pipelines.prepare_market import run_prepare_market
from csi300_multimodal.pipelines.prepare_news import run_prepare_news
from csi300_multimodal.pipelines.run_suite import run_suite
from csi300_multimodal.pipelines.train import run_train


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser.parse_args()


def crawl_news_main() -> None:
    args = _parse_args()
    run_crawl_news(load_config(args.config, args.overrides))


def prepare_market_main() -> None:
    args = _parse_args()
    run_prepare_market(load_config(args.config, args.overrides))


def prepare_news_main() -> None:
    args = _parse_args()
    run_prepare_news(load_config(args.config, args.overrides))


def encode_text_main() -> None:
    args = _parse_args()
    run_encode_text(load_config(args.config, args.overrides))


def build_dataset_main() -> None:
    args = _parse_args()
    run_build_dataset(load_config(args.config, args.overrides))


def train_main() -> None:
    args = _parse_args()
    run_train(load_config(args.config, args.overrides))


def evaluate_main() -> None:
    args = _parse_args()
    run_evaluate(load_config(args.config, args.overrides))


def run_suite_main() -> None:
    args = _parse_args()
    run_suite(load_config(args.config, args.overrides))
