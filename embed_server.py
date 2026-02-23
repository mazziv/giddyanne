"""Shared embedding service. Loads models once, serves all projects.

Start:  .venv/bin/python embed_server.py --daemon
Test:   curl --unix-socket ~/.local/state/giddyanne/embed.sock http://localhost/health
Stop:   .venv/bin/python embed_server.py --stop
"""

import argparse
import asyncio
import atexit
import logging
import os
import signal
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from src.embeddings import LocalEmbedding
from src.global_config import GlobalConfig

logger = logging.getLogger("giddyanne.embed_server")

# model_name -> LocalEmbedding
_models: dict[str, LocalEmbedding] = {}


def _get_model(model_name: str) -> LocalEmbedding:
    """Get or lazy-load a model by name."""
    if model_name not in _models:
        logger.info(f"Loading model: {model_name}")
        provider = LocalEmbedding(model_name)
        # Force load so first request pays the cost here, not mid-embed
        provider.dimension()
        logger.info(f"Model ready: {model_name} (dim={provider.dimension()})")
        _models[model_name] = provider
    return _models[model_name]


class EmbedRequest(BaseModel):
    texts: list[str]
    model: str = "all-MiniLM-L6-v2"


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    dimension: int


class HealthResponse(BaseModel):
    status: str
    models_loaded: list[str]


class ModelInfo(BaseModel):
    dimension: int
    max_seq_length: int


class ModelsResponse(BaseModel):
    models: dict[str, ModelInfo]


# Socket path is set at startup, used by lifespan
_socket_path: str = ""


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: clean stale socket
    if os.path.exists(_socket_path):
        os.unlink(_socket_path)
    yield
    # Shutdown: clean socket
    if os.path.exists(_socket_path):
        os.unlink(_socket_path)


app = FastAPI(lifespan=lifespan)


@app.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest):
    provider = _get_model(req.model)
    embeddings = await provider.embed(req.texts)
    return EmbedResponse(
        embeddings=embeddings,
        dimension=provider.dimension(),
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        models_loaded=list(_models.keys()),
    )


@app.get("/models", response_model=ModelsResponse)
async def models():
    return ModelsResponse(
        models={
            name: ModelInfo(
                dimension=provider.dimension(),
                max_seq_length=provider.max_seq_length,
            )
            for name, provider in _models.items()
        }
    )


def configure_logging(verbose: bool = False, log_file: Path | None = None) -> None:
    """Configure logging level and optional file output."""
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"

    logging.basicConfig(level=level, format=fmt)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(fmt))
        logging.getLogger().addHandler(file_handler)


def write_pid_file(pid_path: Path) -> None:
    """Write current PID to PID file."""
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(os.getpid()))


def remove_pid_file(pid_path: Path) -> None:
    """Remove PID file."""
    pid_path.unlink(missing_ok=True)


def is_running(pid_path: Path) -> int | None:
    """Check if embed server is running. Returns PID or None."""
    if not pid_path.exists():
        return None
    try:
        pid = int(pid_path.read_text().strip())
        os.kill(pid, 0)
        return pid
    except (ValueError, OSError):
        # Stale PID file
        pid_path.unlink(missing_ok=True)
        return None


def spawn_daemon(
    socket_path: str, model: str, verbose: bool = False,
) -> None:
    """Spawn a new background process."""
    import subprocess

    python = sys.executable
    script = Path(__file__).resolve()

    cmd = [python, str(script), "--background",
           "--socket", socket_path, "--model", model]
    if verbose:
        cmd.append("--verbose")

    with open(os.devnull, "w") as devnull:
        subprocess.Popen(
            cmd,
            stdout=devnull,
            stderr=devnull,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Giddyanne shared embedding server")
    parser.add_argument(
        "--daemon", "-d",
        action="store_true",
        help="Run as a background daemon"
    )
    parser.add_argument(
        "--background",
        action="store_true",
        help=argparse.SUPPRESS,  # Internal flag
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop a running daemon"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Check if daemon is running"
    )
    parser.add_argument(
        "--socket",
        type=str,
        default=None,
        help="Unix socket path (default: from global config)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Embedding model name (default: from global config)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = GlobalConfig.load()

    # CLI args override config
    socket_path = args.socket or str(config.socket_path)
    model = args.model or config.embed_model
    pid_path = config.pid_path
    log_path = config.log_path

    # Handle --status
    if args.status:
        pid = is_running(pid_path)
        if pid:
            print(f"Running (PID {pid}, socket {socket_path})")
            # Try to show loaded models
            try:
                import httpx
                with httpx.Client(
                    transport=httpx.HTTPTransport(uds=socket_path), timeout=2.0,
                ) as client:
                    resp = client.get("http://localhost/health")
                    if resp.status_code == 200:
                        data = resp.json()
                        loaded = data.get("models_loaded", [])
                        if loaded:
                            print(f"Models: {', '.join(loaded)}")
            except Exception:
                pass
            sys.exit(0)
        else:
            print("Not running")
            sys.exit(1)

    # Handle --stop
    if args.stop:
        pid = is_running(pid_path)
        if pid is None:
            print("Not running")
            sys.exit(1)
        try:
            os.kill(pid, signal.SIGTERM)
            import time
            for _ in range(30):
                try:
                    os.kill(pid, 0)
                    time.sleep(0.1)
                except OSError:
                    break
            remove_pid_file(pid_path)
            print("Stopped")
            sys.exit(0)
        except OSError as e:
            print(f"Error stopping: {e}", file=sys.stderr)
            sys.exit(1)

    # Check if already running
    existing_pid = is_running(pid_path)
    if existing_pid:
        print(f"Already running (PID {existing_pid})")
        sys.exit(1)

    # Handle --daemon: spawn background process and exit
    if args.daemon:
        spawn_daemon(socket_path, model, args.verbose)
        print(f"Embed server starting (socket {socket_path})")
        sys.exit(0)

    # From here: running the actual server (foreground or --background)

    # Set module-level socket path for lifespan
    _socket_path = socket_path

    configure_logging(args.verbose, log_path)

    # Write PID file and register cleanup
    write_pid_file(pid_path)
    atexit.register(remove_pid_file, pid_path)

    # Become process group leader
    try:
        os.setpgrp()
    except OSError:
        pass

    # Pre-warm the default model
    logger.info(f"Pre-warming model: {model}")
    _get_model(model)

    # Signal handlers - kill entire process group
    def handle_signal(signum, frame):
        remove_pid_file(pid_path)
        if os.path.exists(socket_path):
            try:
                os.unlink(socket_path)
            except OSError:
                pass
        os.killpg(os.getpid(), signal.SIGKILL)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    try:
        uvicorn.run(
            app,
            uds=socket_path,
            log_level="debug" if args.verbose else "info",
        )
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    except Exception as e:
        print(f"Startup failed: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        remove_pid_file(pid_path)
