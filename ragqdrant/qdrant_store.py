from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from .exceptions import QdrantError


class QdrantStore:
    def __init__(
        self,
        url: str,
        api_key: Optional[str],
        prefer_grpc: bool,
        timeout_sec: int,
    ) -> None:
        try:
            self.client = QdrantClient(url=url, api_key=api_key, prefer_grpc=prefer_grpc, timeout=timeout_sec)
        except Exception as e:
            raise QdrantError(f"Failed to initialize Qdrant client: {e}")

    def ensure_collection(
        self,
        name: str,
        vector_size: int,
        distance: str,
        hnsw_m: int,
        hnsw_ef_construct: int,
        optimizers: Dict[str, Any],
        replication_factor: int,
        write_consistency_factor: int,
        shards_number: int,
    ) -> None:
        try:
            distance_enum = getattr(rest.Distance, distance)
        except Exception:
            distance_enum = rest.Distance.COSINE
        try:
            exists = self.client.collection_exists(name)
            if not exists:
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=rest.VectorParams(
                        size=vector_size, distance=distance_enum, hnsw_config=rest.HnswConfigDiff(m=hnsw_m, ef_construct=hnsw_ef_construct)
                    ),
                    optimizers_config=rest.OptimizersConfigDiff(**optimizers),
                    replication_factor=replication_factor,
                    write_consistency_factor=write_consistency_factor,
                    shards_number=shards_number,
                )
        except Exception as e:
            raise QdrantError(f"Failed to ensure collection {name}: {e}")

    def upsert_points(
        self,
        collection: str,
        ids: Sequence[str],
        vectors: np.ndarray,
        payloads: Sequence[Dict[str, Any]],
        timeout: Optional[int] = None,
    ) -> None:
        try:
            points = []
            for pid, vec, payload in zip(ids, vectors.tolist(), payloads):
                points.append(rest.PointStruct(id=pid, vector=vec, payload=payload))
            self.client.upsert(collection_name=collection, points=points, timeout=timeout)
        except Exception as e:
            raise QdrantError(f"Failed to upsert points: {e}")

    def search_similar(
        self,
        collection: str,
        vector: np.ndarray,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        filter_: Optional[rest.Filter] = None,
    ) -> List[rest.ScoredPoint]:
        try:
            res = self.client.search(
                collection_name=collection,
                query_vector=vector.tolist(),
                limit=top_k,
                score_threshold=score_threshold,
                query_filter=filter_,
            )
            return res
        except Exception as e:
            raise QdrantError(f"Search failed: {e}")