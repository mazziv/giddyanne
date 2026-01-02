# Configuration

Giddyanne requires a `.giddyanne.yaml` file in your project root.

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

The database is stored at `.giddyanne/<model-name>/vectors.lance`. Each model gets its own index, so you can switch models without losing data.

Available models:
- `all-MiniLM-L6-v2` (default) - Good balance of speed and quality
- `all-mpnet-base-v2` - Higher quality, slower
- `paraphrase-MiniLM-L3-v2` - Faster, lower quality
