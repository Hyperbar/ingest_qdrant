from __future__ import annotations

import threading
from typing import Optional

from prometheus_client import Counter, Gauge, Histogram, start_http_server

INGESTED_POINTS = Counter("ingested_points_total", "Total ingested points into Qdrant")
INGEST_ERRORS = Counter("ingest_errors_total", "Total errors during ingestion")
EMBEDDING_TIME = Histogram("embedding_time_seconds", "Time spent computing embeddings")
CHUNK_COUNT = Counter("chunks_total", "Total chunks produced")
BATCH_LATENCY = Histogram("batch_upsert_latency_seconds", "Time to upsert a batch to Qdrant")


class MetricsServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 8008) -> None:
        self.host = host
        self.port = port
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=start_http_server, args=(self.port,), daemon=True)
        self._thread.start()