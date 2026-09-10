# LEDGER Docker launch

Docker images install dependencies at build time only. Every container start
runs an offline requirement-version check and `pip check`; it never runs
`pip install`. Docker BuildKit caches downloaded wheels between rebuilds.

## Full stack (default)

This starts all seven services. Retrieval uses the real FAISS/BM25/reranking
pipeline and uploaded PDFs are processed by the Docling backend.

```powershell
docker compose build
docker compose up -d
docker compose ps
docker compose logs -f
```

Open <http://localhost:8007>. Stop without deleting caches:

```powershell
docker compose down
```

After the first build, normal launches only need:

```powershell
docker compose up -d
```

## First build (large download)

The first build downloads CPU-only PyTorch, Docling, OCR, embedding, and
reranking dependencies. Docker BuildKit caches package downloads and image
layers. The Hugging Face/Docling, RapidOCR, and processed-document caches
survive container replacement in named volumes.

```powershell
docker compose build
docker compose up -d
```

## Rebuild rules

- Source-only change: `docker compose build <service>` reuses dependency layers.
- Requirements change: the affected dependency layer is rebuilt using the
  BuildKit pip cache.
- Start/restart: no dependency installation occurs.
- Do not use `docker compose down -v` unless you intentionally want to delete
  downloaded model and processed-document caches.
