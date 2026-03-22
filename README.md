<<<<<<< HEAD
# paper-code
=======
# CSI300 Multimodal Research Pipeline

This repository now includes a production-style news crawling and preprocessing pipeline for a CSI300 finance thesis workflow. The news system is designed to preserve raw articles, standardize article-level text, enforce anti-lookahead trading-day alignment, and generate daily text features that feed the existing modeling pipeline.

## Architecture

The project is organized as a staged data pipeline:

- `crawl-news`: crawl configurable sources from `configs/sources.yaml` and write raw article records to `data/raw/news/raw_articles.parquet`
- `prepare-news`: load raw or legacy news input, clean text, deduplicate, align to trading dates, and build daily placeholder features
- `encode-text`: run the configured text encoder on aligned article-level news
- `build-dataset`: merge daily text features into the market table and update placeholder sentiment and embedding outputs when encoded article news exists

Key source locations:

- `src/csi300_multimodal/crawler/`: crawling infrastructure and source adapters
- `src/csi300_multimodal/preprocess/`: cleaning, deduplication, trade-date alignment, and daily feature generation
- `src/csi300_multimodal/pipelines/`: CLI-facing pipeline stages
- `configs/sources.yaml`: source registry
- `scripts/run_news_pipeline_example.py`: offline demo runner using fixture HTML pages

## Output Layout

News outputs are explicit and reproducible by stage:

- `data/raw/news/raw_articles.parquet`
- `data/processed/news/cleaned_articles.parquet`
- `data/processed/news/deduped_articles.parquet`
- `data/processed/news/aligned_articles.parquet`
- `data/processed/news/daily_text_features.parquet`
- `data/processed/news/encoded_news.parquet`
- `data/processed/news/daily_embeddings.parquet`
- `data/processed/main_table.parquet`

Raw article storage preserves the original crawl payload. Processed outputs add cleaned text fields, duplicate audit fields, `trade_date`, and daily aggregated text features.

## Article Schema

The unified article schema includes:

- `news_id`
- `source`
- `url`
- `title`
- `publish_time`
- `content`
- `crawl_time`
- `channel`

Processed article tables also include derived fields such as `raw_title`, `raw_content`, `title_clean`, `content_clean`, `title_hash`, `content_hash`, `is_duplicate`, `duplicate_reason`, `canonical_news_id`, and `trade_date`.

## Anti-Lookahead Trade-Date Alignment

Trading-date assignment uses the market file as the trading calendar source of truth.

- If an article is published before `15:00:00` on a trading day, it is assigned to that same trading day.
- If it is published at `15:00:00` or later, it is assigned to the next trading day.
- If it is published on a weekend or market holiday, it is assigned to the next trading day.

This enforces a strict no-lookahead rule for downstream daily forecasting.

## Daily Text Features

`prepare-news` builds daily features directly from aligned, deduplicated article news:

- `news_count`
- `has_news`
- `title_len_mean`, `title_len_median`, `title_len_min`, `title_len_max`
- `content_len_mean`, `content_len_median`, `content_len_min`, `content_len_max`
- placeholder sentiment columns: `finbert_pos_mean`, `finbert_neg_mean`, `finbert_neu_mean`, `sentiment_score`, `sentiment_abs`

The placeholder sentiment values are neutral until `encode-text` is run. After `encode-text`, `build-dataset` recomputes daily sentiment aggregates from article-level encoded outputs and overwrites the placeholder sentiment columns. `daily_embeddings.parquet` is also overwritten with pooled article embeddings when available.

## Running The Pipeline

Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

Run the news and modeling pipeline step by step:

```bash
prepare-market --config configs/default.yaml
crawl-news --config configs/default.yaml
prepare-news --config configs/default.yaml
encode-text --config configs/default.yaml
build-dataset --config configs/default.yaml
train --config configs/default.yaml
evaluate --config configs/default.yaml
```

Run the offline example:

```bash
python scripts/run_news_pipeline_example.py
```

The example uses local fixture HTML pages under `data/raw/news/example_source/`, so it works without live website access.

## Adding New Sources

Add a new entry in `configs/sources.yaml` with:

- `name`
- `enabled`
- `adapter`
- `start_urls`
- `request`
- `crawl`
- `parser`

For HTML sources that follow a list-detail structure, you can subclass `HtmlListDetailSourceAdapter` in `src/csi300_multimodal/crawler/sources/base.py` and register the adapter in `src/csi300_multimodal/crawler/run_crawler.py`.

Start with a source-specific adapter when:

- listing pages need custom pagination logic
- publish-time parsing is non-standard
- content extraction needs source-specific cleanup
- the site requires JavaScript rendering

Playwright is intentionally not used in v1. Add it later only if a real source cannot be handled with `requests` and `BeautifulSoup`.

## Legacy News Input

If you already have a pre-exported news table, `prepare-news` will fall back to `data.news_path` when `data.raw_news_path` does not exist. Legacy tables only need `publish_time`, `title`, and `content`; missing schema fields are filled deterministically.

## Tests

The repository includes coverage for:

- crawler parsing
- cleaning and deduplication
- anti-lookahead alignment
- end-to-end news pipeline smoke execution

Run:

```bash
pytest
```
>>>>>>> 4503937 (initial commit)
