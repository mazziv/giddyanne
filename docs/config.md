# Configuration

Giddyanne works in any git repo without configuration. A `.giddyanne.yaml` file is optional but recommended for better search quality.

## Zero-Config Mode

In a git repo without `.giddyanne.yaml`, giddyanne automatically:
- Uses the git root as the project root
- Indexes common directories (`src/`, `lib/`, `app/`, `tests/`, `docs/`) or the whole project if none exist
- Respects `.gitignore` patterns
- Stores indexes in the system temp directory (`$TMPDIR/giddyanne/<name>-<hash>/`)

Indexes rebuild automatically if the temp directory gets cleaned. This is fine — `giddy up` is fast.

## Config File

For better search quality, add a `.giddyanne.yaml` to your project root. Path descriptions are embedded alongside content, improving semantic matching.

## Minimal Config

```yaml
paths:
  - path: .
    description: My app
```

That's it. Only supported language files are indexed, and `.gitignore` is respected automatically.

## Annotated Example

```yaml
paths:
  - path: src/
    description: Core application source code

  - path: src/auth/
    description: Authentication, login, sessions, permissions

  - path: tests/
    description: Test suite and fixtures
```

Descriptions are embedded alongside content for better semantic matching. More specific descriptions improve search quality.

## Generating a Config

Run `giddy init` to get a prompt you can paste into Claude or another LLM:

```bash
claude "follow the instructions: $(giddy init)"
```

The LLM will analyze your codebase and generate an appropriate config.

## Full Example

All available options:

```yaml
paths:
  # Index the entire project with a general description
  - path: .
    description: Web application with REST API

  # Override with more specific descriptions for key directories
  - path: src/api/
    description: REST endpoints, request handlers, middleware

  - path: src/auth/
    description: Authentication, JWT tokens, session management, permissions

  - path: src/db/
    description: Database models, migrations, query builders

  - path: src/utils/
    description: Shared utilities, helpers, validation functions

  - path: tests/
    description: Test suite, fixtures, mocks

settings:
  # File handling
  max_file_size: 1000000           # Skip files > 1MB
  ignore_patterns:                  # Beyond .gitignore
    - "*.generated.*"
    - "*.min.js"
    - "vendor/"

  # Server
  host: "0.0.0.0"                  # Bind address
  port: 8000                        # HTTP port

  # Chunking (defaults work well for most codebases)
  min_chunk_lines: 10
  max_chunk_lines: 50
  overlap_lines: 5

  # Embedding model
  local_model: "all-MiniLM-L6-v2"

  # Ollama backend (optional, disabled by default)
  ollama: true
  ollama_url: "http://localhost:11434"
  ollama_model: "all-minilm:l6-v2"

  # Reranker (optional, disabled by default)
  reranker_model: "cross-encoder/ms-marco-MiniLM-L6-v2"
```

## Settings Reference

### Paths

| Field | Required | Description |
|-------|----------|-------------|
| `path` | Yes | File or directory to index (relative to config file) |
| `description` | No | Human-readable description, embedded alongside content for better search |

### File Handling

| Setting | Default | Description |
|---------|---------|-------------|
| `max_file_size` | 1000000 | Skip files larger than this (bytes) |
| `ignore_patterns` | [] | Additional patterns to ignore (uses gitignore syntax) |

### Server

| Setting | Default | Description |
|---------|---------|-------------|
| `host` | "0.0.0.0" | Network interface to bind to |
| `port` | 8000 | HTTP port for the server |

### Chunking

Code is split into chunks for indexing. These settings control chunk boundaries:

| Setting | Default | Description |
|---------|---------|-------------|
| `min_chunk_lines` | 10 | Minimum lines per chunk |
| `max_chunk_lines` | 50 | Maximum lines per chunk |
| `overlap_lines` | 5 | Lines of overlap between chunks |

Defaults work well for most codebases. Smaller chunks give more precise results but may lose context.

### Embedding

| Setting | Default | Description |
|---------|---------|-------------|
| `local_model` | "all-MiniLM-L6-v2" | Sentence-transformers model name |

With a config file, the database is stored at `.giddyanne/<model-name>/vectors.lance`. In zero-config mode, it's stored in the system temp directory. Each model gets its own index, so you can switch models without losing data.

Available models:
- `all-MiniLM-L6-v2` (default) — Good balance of speed and quality
- `all-mpnet-base-v2` — Higher quality, slower
- `paraphrase-MiniLM-L3-v2` — Faster, lower quality
- `nomic-ai/CodeRankEmbed` — Code-specialized (768d), but 5-6x slower indexing with no measurable quality gain. Unusable on laptop hardware for repos over ~100 files.

### Ollama

Use a local [Ollama](https://ollama.com) server for GPU-accelerated embeddings instead of the default sentence-transformers (CPU).

| Setting | Default | Description |
|---------|---------|-------------|
| `ollama` | false | Enable Ollama backend |
| `ollama_url` | "http://localhost:11434" | Ollama server URL |
| `ollama_model` | "all-minilm:l6-v2" | Ollama embedding model |

```yaml
settings:
  ollama: true
  ollama_model: "all-minilm:l6-v2"
```

Without a config file, use the `--ollama` CLI flag instead:

```bash
giddy up --ollama
```

For MCP, set the `GIDDY_OLLAMA=1` environment variable in your MCP config.

Requires Ollama running locally with the model pulled (`ollama pull all-minilm:l6-v2`).

### Reranker

| Setting | Default | Description |
|---------|---------|-------------|
| `reranker_model` | "" (disabled) | Cross-encoder model for result reranking |

The reranker is a second-stage scorer that re-evaluates the top search results using a cross-encoder. It improves ranking quality at the cost of slightly slower queries (the model scores each result individually).

```yaml
settings:
  reranker_model: "cross-encoder/ms-marco-MiniLM-L6-v2"
```

Without a config file, use the `--rerank` CLI flag instead:

```bash
giddy up --rerank
```

For MCP, set the `GIDDY_RERANK=1` environment variable in your MCP config.

## Global Config

The shared embedding server has its own config at `~/.config/giddyanne/config.yaml`. This is separate from per-project `.giddyanne.yaml` files.

```yaml
# All fields are optional — defaults shown
embed_model: "all-MiniLM-L6-v2"
auto_start: true
socket_path: "~/.local/state/giddyanne/embed.sock"
pid_path: "~/.local/state/giddyanne/embed.pid"
log_path: "~/.local/state/giddyanne/embed.log"
```

| Setting | Default | Description |
|---------|---------|-------------|
| `embed_model` | "all-MiniLM-L6-v2" | Model the shared server loads |
| `auto_start` | true | Auto-start the embed server when a project server starts |
| `socket_path` | `~/.local/state/giddyanne/embed.sock` | Unix socket for embed server |
| `pid_path` | `~/.local/state/giddyanne/embed.pid` | PID file location |
| `log_path` | `~/.local/state/giddyanne/embed.log` | Log file location |

Set `auto_start: false` to disable the shared server and fall back to in-process embedding (each project loads its own model copy).
