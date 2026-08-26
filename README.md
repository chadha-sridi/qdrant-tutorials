# Multi-Stage Retrieval With Qdrant

This example builds a retrieval pipeline with three stages:

1. Dense retrieval with `BAAI/bge-base-en-v1.5` finds an initial candidate set.
2. A Qdrant formula boosts candidates using citation count and publication year.
3. ColBERT late-interaction reranking selects the final results.

The collection uses named vectors for the dense and ColBERT representations. Payload indexes support filtering by `user_id`, `paper_id`, `citation_count`, and `published_year`.

## Setup

The commands below can be run from the repository root:

```bash
cd multi-stage_retrieval
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Set `QDRANT_URL` and `QDRANT_API_KEY` in `.env`. The embedding models are downloaded on first use.

## Run

Ingest the example paper. The script creates the collection and payload indexes automatically when they do not already exist:

```bash
	python ingest.py
```

Run the multi-stage search after ingestion:

```bash
	python multi-stage_retrieval.py
```

The example uses `demo_user` and paper `1810.04805`. Change those values in the `__main__` blocks to ingest or search different data.

## Notes

- The Semantic Scholar API enriches each paper with citation count and publication year. If that request fails, ingestion continues with default metadata.
- The collection name is `demo_collection1`, and rerunning ingestion adds new points with new IDs.