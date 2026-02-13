"""Shared setup for HTTP and MCP entry points."""

import logging
import os
import signal
from dataclasses import dataclass
from pathlib import Path

from .embeddings import EmbeddingProvider, EmbeddingService
from .engine import (
    FileIndexer,
    IndexingProgress,
    StatsTracker,
    create_embedding_provider,
)
from .project_config import FileFilter, ProjectConfig
from .vectorstore import VectorStore
from .watcher import FileWatcher

logger = logging.getLogger(__name__)


def get_db_path(storage_dir: Path, model_name: str) -> Path:
    """Get database path with model name included.

    Returns e.g. '<storage_dir>/all-MiniLM-L6-v2/vectors.lance'
    """
    safe_model = model_name.replace("/", "-")
    return storage_dir / safe_model / "vectors.lance"


@dataclass
class AppComponents:
    project_config: ProjectConfig
    storage_dir: Path
    embedding_service: EmbeddingService
    vector_store: VectorStore
    file_filter: FileFilter
    indexer: FileIndexer
    stats: StatsTracker
    progress: IndexingProgress
    provider: EmbeddingProvider


async def create_components(root_path: Path, project_config: ProjectConfig,
                     storage_dir: Path, log_path: Path | None = None,
                     log_filename: str = "server.log") -> AppComponents:
    """Load config, initialize all components, connect to DB.

    If log_path is None, it's derived from the resolved db_path using log_filename.
    """
    provider = create_embedding_provider(project_config)
    embedding_service = EmbeddingService(provider)

    db_path = get_db_path(storage_dir, provider.model_name)
    vector_store = VectorStore(db_path, provider.dimension())
    await vector_store.connect()

    if log_path is None:
        log_path = db_path.parent / log_filename

    stats = StatsTracker(log_path, root_path)
    progress = IndexingProgress()

    file_filter = FileFilter(root_path, project_config)

    indexer = FileIndexer(
        embedding_service,
        vector_store,
        file_filter,
        stats,
        progress,
    )

    return AppComponents(
        project_config=project_config,
        storage_dir=storage_dir,
        embedding_service=embedding_service,
        vector_store=vector_store,
        file_filter=file_filter,
        indexer=indexer,
        stats=stats,
        progress=progress,
        provider=provider,
    )


async def run_indexing(components: AppComponents) -> None:
    """Reconcile stale files, run full index."""
    removed = await components.indexer.reconcile_index()
    if removed:
        logger.info(f"Removed {removed} stale files from index")

    await components.indexer.full_index()


async def start_watcher(components: AppComponents) -> FileWatcher:
    """Start file watcher for incremental updates."""
    watcher = FileWatcher(components.file_filter, components.indexer.handle_event)
    await watcher.start()
    logger.info("File watcher started")
    return watcher


def setup_signal_handlers():
    """Process group leader + SIGTERM/SIGINT handlers."""
    try:
        os.setpgrp()
    except OSError:
        pass  # Already a process group leader

    def handle_signal(signum, frame):
        os.killpg(os.getpid(), signal.SIGKILL)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
