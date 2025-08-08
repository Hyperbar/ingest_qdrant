from __future__ import annotations

import glob
from pathlib import Path
from typing import List, Optional

import typer

from ragqdrant.config import load_config
from ragqdrant.logging_config import configure_logging
from ragqdrant.metrics import MetricsServer
from ragqdrant.pipeline import IngestionPipeline, FileLoader

app = typer.Typer(help="RAG -> Qdrant ingestion pipeline")


@app.command()
def run(
    input_dir: str = typer.Option(..., help="Directory to scan for documents"),
    collection: str = typer.Option(..., help="Qdrant collection name"),
    config: str = typer.Option("./config/pipeline.yaml", help="Path to YAML configuration"),
    pattern: str = typer.Option("**/*.*", help="Glob pattern for files"),
) -> None:
    configure_logging()
    cfg = load_config(config)

    if cfg.metrics.enabled:
        MetricsServer(host=cfg.metrics.host, port=cfg.metrics.port).start()

    all_paths = [
        p
        for p in glob.glob(str(Path(input_dir) / pattern), recursive=True)
        if Path(p).is_file() and Path(p).suffix.lower() in FileLoader.SUPPORTED_EXTENSIONS
    ]
    if not all_paths:
        typer.secho("No matching files found.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=0)

    pipe = IngestionPipeline(cfg)
    pipe.ingest_paths(all_paths, collection)


if __name__ == "__main__":
    app()