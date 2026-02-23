"""Embed server lifecycle helpers. Used by project servers and the Go CLI."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from src.global_config import GlobalConfig

logger = logging.getLogger(__name__)


def is_embed_server_running(config: GlobalConfig) -> int | None:
    """Check if embed server process is alive. Returns PID or None."""
    if not config.pid_path.exists():
        return None
    try:
        pid = int(config.pid_path.read_text().strip())
        os.kill(pid, 0)
        return pid
    except (ValueError, OSError):
        config.pid_path.unlink(missing_ok=True)
        return None


def is_embed_server_healthy(config: GlobalConfig) -> bool:
    """Check if embed server responds to health check on its Unix socket."""
    import httpx

    socket_path = str(config.socket_path)
    if not os.path.exists(socket_path):
        return False

    try:
        with httpx.Client(
            transport=httpx.HTTPTransport(uds=socket_path), timeout=2.0,
        ) as client:
            resp = client.get("http://localhost/health")
            return resp.status_code == 200
    except Exception:
        return False


def start_embed_server(config: GlobalConfig, verbose: bool = False) -> bool:
    """Spawn embed_server.py --daemon and poll until healthy. Returns True on success."""
    # Find the embed_server.py script relative to this file
    embed_script = Path(__file__).resolve().parent.parent / "embed_server.py"
    if not embed_script.exists():
        logger.warning(f"embed_server.py not found at {embed_script}")
        return False

    python = sys.executable
    cmd = [python, str(embed_script), "--daemon",
           "--socket", str(config.socket_path),
           "--model", config.embed_model]
    if verbose:
        cmd.append("--verbose")

    logger.info("Starting shared embed server...")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to start embed server: {e.stderr.strip()}")
        return False

    # Poll for health (model loading can take a few seconds)
    for _ in range(60):
        time.sleep(0.5)
        if is_embed_server_healthy(config):
            logger.info("Shared embed server is ready")
            return True

    logger.warning("Embed server started but never became healthy")
    return False


def stop_embed_server(config: GlobalConfig) -> bool:
    """Stop the embed server. Returns True if stopped."""
    pid = is_embed_server_running(config)
    if pid is None:
        return False
    try:
        os.kill(pid, signal.SIGTERM)
        for _ in range(30):
            try:
                os.kill(pid, 0)
                time.sleep(0.1)
            except OSError:
                break
        config.pid_path.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def ensure_embed_server(config: GlobalConfig | None = None) -> bool:
    """Ensure embed server is running and healthy. Start if needed.

    Returns True if the server is healthy (already running or just started).
    Returns False if auto_start is disabled or startup failed.
    """
    if config is None:
        config = GlobalConfig.load()

    if is_embed_server_healthy(config):
        return True

    if not config.auto_start:
        return False

    return start_embed_server(config)
