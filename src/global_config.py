"""Global config for shared services (~/.config/giddyanne/config.yaml)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def _default_state_dir() -> Path:
    return Path.home() / ".local" / "state" / "giddyanne"


@dataclass
class GlobalConfig:
    """Global configuration for the shared embedding service."""

    socket_path: Path = field(default_factory=lambda: _default_state_dir() / "embed.sock")
    pid_path: Path = field(default_factory=lambda: _default_state_dir() / "embed.pid")
    log_path: Path = field(default_factory=lambda: _default_state_dir() / "embed.log")
    embed_model: str = "all-MiniLM-L6-v2"
    auto_start: bool = True

    @classmethod
    def load(cls) -> GlobalConfig:
        """Load from ~/.config/giddyanne/config.yaml, or return defaults."""
        config_path = Path.home() / ".config" / "giddyanne" / "config.yaml"
        if not config_path.exists():
            return cls()

        import yaml

        with open(config_path) as f:
            data = yaml.safe_load(f) or {}

        state_dir = _default_state_dir()
        kwargs: dict = {}

        if "socket_path" in data:
            kwargs["socket_path"] = Path(data["socket_path"]).expanduser()
        if "pid_path" in data:
            kwargs["pid_path"] = Path(data["pid_path"]).expanduser()
        if "log_path" in data:
            kwargs["log_path"] = Path(data["log_path"]).expanduser()
        if "embed_model" in data:
            kwargs["embed_model"] = data["embed_model"]
        if "auto_start" in data:
            kwargs["auto_start"] = bool(data["auto_start"])

        # If state_dir is set, use it as base for any paths not explicitly set
        if "state_dir" in data:
            state_dir = Path(data["state_dir"]).expanduser()
            kwargs.setdefault("socket_path", state_dir / "embed.sock")
            kwargs.setdefault("pid_path", state_dir / "embed.pid")
            kwargs.setdefault("log_path", state_dir / "embed.log")

        return cls(**kwargs)
