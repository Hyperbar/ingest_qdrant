# RAG + Qdrant Ingestion Pipeline

This repository provides a production-ready ingestion pipeline for Retrieval-Augmented Generation (RAG) into Qdrant, featuring adaptive chunking, metadata enrichment, deduplication, and high-throughput batching.

## Features
- Adaptive and hierarchical chunking with token-aware sizing and overlaps
- Rich metadata extraction (language, entities, keywords, topics placeholder)
- Deduplication (exact and near-duplicate via similarity checks)
- Embedding via Sentence-Transformers (BGE, E5, multilingual)
- Qdrant collection auto-provisioning with optimized HNSW config
- Batched streaming ingestion with structured logging and Prometheus metrics
- Config-driven (YAML) with strict schema validation
- CLI via Typer for easy operation

## Requirements
- Python 3.10+
- A running Qdrant instance (Cloud or self-hosted)

## Quickstart
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Configure pipeline in `config/pipeline.yaml`.
3. Run ingestion:
   ```bash
   python ingest.py \
     --input-dir ./sample_docs \
     --collection my_docs \
     --config ./config/pipeline.yaml \
     --pattern "**/*.{txt,md,pdf,docx,html}"
   ```

## Configuration
See `config/pipeline.yaml` for all options (Qdrant, embeddings, chunking, metadata, deduplication, batching).

## Monitoring
- Prometheus metrics exposed on `0.0.0.0:8008/metrics`
- Structured JSON logging with context-rich events

## Testing
```bash
pytest -q
```

## Notes
- PDF parsing provided via `pypdf` (text only). For complex PDFs (tables/images), integrate additional extractors.
- Optional spaCy NER can be enabled if a model is installed (e.g., `en_core_web_sm`).
