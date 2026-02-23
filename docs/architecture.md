# Architecture

How giddyanne works internally. Read this before changing the indexing pipeline, search logic, or server lifecycle.

For user-facing docs, see the [home page](index.md). For dev setup, see [Contributing](contributing.md).

## System Overview

```
┌─────────────┐     HTTP      ┌──────────────────────────────────┐
│  Go CLI     │──────────────▶│  FastAPI Server (http_main.py)   │
│  (giddy)    │               │                                  │
└─────────────┘               │  ┌────────┐ ┌────────┐           │
                              │  │ engine │ │chunker │           │
┌─────────────┐  stdio        │  └───┬────┘ └────────┘           │
│ Claude Code │──────────────▶│      │                           │
│  (MCP)      │               │  ┌───▼────────────┐              │
└─────────────┘               │  │ LanceDB        │              │
                              │  │ (vectorstore)  │              │
                              │  └────────────────┘              │
                              │  ┌────────────────┐              │
                              │  │ watchdog       │              │
                              │  │ (file watcher) │              │
                              │  └────────────────┘              │
                              └───────────┬──────────────────────┘
                                          │ Unix socket
                              ┌───────────▼──────────────────────┐
                              │  Embed Server (embed_server.py)  │
                              │  Global singleton, one per machine│
                              │  ┌──────────────────────────┐    │
                              │  │ sentence-transformers     │    │
                              │  │ (loads model once ~727MB) │    │
                              │  └──────────────────────────┘    │
                              └──────────────────────────────────┘
```

Three entry points into the Python core:

- **`http_main.py`** starts an HTTP server per project. The Go CLI talks to it over HTTP.
- **`mcp_main.py`** starts an MCP server over stdio. Claude Code talks to it directly.
- **`embed_server.py`** is a global singleton that loads the embedding model once and serves all project servers over a Unix socket (`~/.local/state/giddyanne/embed.sock`). Auto-started by project servers, or managed directly via `giddy embed start|stop|status`.

Project servers use `SharedEmbedding` to proxy embed calls to the shared server instead of loading the ~900MB model in-process. This means running 3 projects costs ~727MB (one model) + 3×~180MB (project servers) instead of 3×~900MB. Falls back to in-process `LocalEmbedding` if the shared server is unavailable.

Both `http_main.py` and `mcp_main.py` call `src/startup.py` to initialize shared components (config, embeddings, vectorstore, indexer), then add their transport layer on top.

## Source Map

All Python source lives in `src/`. Each file has one job:

| File | What it does |
|------|-------------|
| `engine.py` | Orchestrates indexing — file discovery, mtime checks, batch processing. Start here. |
| `chunker.py` | Splits files into chunks using language-aware separators |
| `embeddings.py` | Embedding providers (local sentence-transformers, shared server, Ollama) |
| `embed_lifecycle.py` | Start/stop/health-check the shared embedding server |
| `global_config.py` | Global config loader for `~/.config/giddyanne/config.yaml` |
| `vectorstore.py` | All LanceDB operations — upsert, delete, search, caching |
| `watcher.py` | Watches the filesystem for changes, debounces events |
| `api.py` | FastAPI endpoints (`/search`, `/status`, `/stats`, `/health`) |
| `mcp_server.py` | MCP tool definitions for Claude Code |
| `startup.py` | Shared initialization — creates components, runs indexing, signal handlers |
| `project_config.py` | Loads `.giddyanne.yaml`, applies file filters, handles git-only mode |
| `languages.py` | Language definitions — extensions and chunk separators for 24 languages |
| `errors.py` | Custom exceptions |

The Go CLI is a single file: `cmd/giddy/main.go`.

## The Indexing Pipeline

This is the core of the system. It runs on startup and again whenever files change.

### 1. File Discovery

The engine walks configured paths using `rglob("*")`. Each candidate goes through `FileFilter.should_include()` in `project_config.py`:

- Does it have a supported extension? (defined in `languages.py`)
- Is it under the max file size? (default: 1MB)
- Does it match a `.gitignore` pattern? (auto-loaded)
- Does it match any configured `ignore_patterns`?

Files that pass all checks become indexing candidates.

### 2. Mtime Comparison

Before reading or embedding anything, the engine does a single bulk query to load all stored mtimes from the database (`get_all_mtimes()`). Each file's current mtime is compared against what's stored. Unchanged files are skipped entirely.

This is the main performance optimization. After the first index, subsequent startups only process files that actually changed.

### 3. Chunking

Files are split into pieces in `chunker.py`. The chunker uses tree-sitter AST parsing to split code at real node boundaries — functions, classes, and other top-level definitions. Falls back to blank-line splitting for languages without a tree-sitter grammar.

After splitting:
1. **Small chunks get merged** — consecutive chunks under `min_lines` (default: 10) are combined
2. **Large chunks get split** — chunks over `max_lines` (default: 50) are broken up with `overlap_lines` (default: 5) lines of overlap
3. **Token-aware splitting** — after line-based chunking, oversized chunks are recursively halved until they fit the model's token budget (256 tokens for MiniLM). This eliminates silent embedding truncation.
4. **Context enrichment** — each chunk gets file path and function/class name prepended before embedding. The stored content stays raw (search results show code only).

### 4. Embedding

Each chunk gets **three embeddings** (384-dimensional vectors from `all-MiniLM-L6-v2`):

- **path_embedding** — the file path. Helps queries like "tests for auth" find test files.
- **content_embedding** — the actual code. This is the primary search signal.
- **description_embedding** — from the config's path description (e.g., "Authentication and session management"). Gives the model extra context about what this code is for.

Chunks are batched (32 per batch) and embedded in a single model call. File reads are parallelized with an asyncio semaphore (16 concurrent). The goal is to minimize the number of model invocations — that's where the time goes.

### 5. Storage

Everything is upserted into LanceDB. The schema per row:

```
path, chunk_index, start_line, end_line, content, fts_content, description, mtime,
path_embedding[384], content_embedding[384], description_embedding[384]
```

Each file produces multiple rows (one per chunk). When a file is re-indexed, all its existing rows are deleted first, then the new chunks are inserted.

## How Search Works

Search lives in `vectorstore.py`. There are three modes:

**Semantic** — embeds the query, does vector similarity search against content embeddings, and blends in description embedding scores. The blend is `(1 - desc_weight) * content_score + desc_weight * desc_score` with a default description weight of 0.3.

**Full-text** — BM25 keyword search via LanceDB's built-in FTS index. The index uses an enriched `fts_content` column that repeats file path and symbol name 3x before raw content, simulating BM25F field-level boosting. Pure keyword matching with stemming and stop word removal.

**Hybrid** (default) — runs both, then combines results using Reciprocal Rank Fusion (RRF) with tunable weights (semantic 1.2, FTS 0.5). Results are biased by file category (source 1.0x, tests 0.8x, docs 0.6x) and deduplicated by file (best chunk per file). Falls back to semantic-only if the FTS index isn't available yet.

### Query Caching

Query embeddings are cached in a separate LanceDB table. If you search for "auth logic" twice, the second time skips the embedding step entirely. The cache tracks hit counts and last-used timestamps.

## File Watching

The watcher (`watcher.py`) uses the `watchdog` library to monitor configured paths for filesystem events.

Key behaviors:
- **Debouncing** — events are coalesced over a 100ms window. If you save a file three times in quick succession, it only re-indexes once.
- **Event precedence** — DELETE > CREATED > MODIFIED. If a file is created and then deleted within the debounce window, only the delete fires.
- **Filtering** — events are checked against `FileFilter.matches_path()` before processing. Changes to ignored files are silently dropped.
- **Async bridge** — watchdog is synchronous, so the watcher bridges events into the async event loop for the indexer callback.

When a file change fires:
1. Delete all existing chunks for that path
2. Re-read the file
3. Re-chunk with language-aware splitting
4. Re-embed all chunks
5. Upsert to the database

For deletes, it just removes all chunks for that path.

## Startup Sequence

### HTTP Server (`http_main.py`)

1. Parse CLI args (`--daemon`, `--port`, `--host`, `--verbose`, `--path`)
2. Load config from `.giddyanne.yaml` or detect git-only mode
3. Determine storage directory
4. Create embedding provider (shared embed server > in-process sentence-transformers > Ollama). The shared server is auto-started if not running and `auto_start` is enabled.
5. Initialize vectorstore (LanceDB)
6. Start FastAPI server
7. **Reconcile** — scan the database, remove entries for files that no longer exist or are now excluded
8. **Index** — discover files, skip unchanged (mtime), index the rest
9. Start file watcher

In daemon mode, the server spawns a new process with `subprocess.Popen` (not fork — sentence-transformers uses threading internally, and threading + fork don't mix). It writes a PID file and redirects output to a log file.

### MCP Server (`mcp_main.py`)

Same initialization, but runs indexing synchronously on startup (needs the index ready before Claude starts querying) and exposes MCP tools over stdio instead of HTTP.

## The Go CLI

The CLI (`cmd/giddy/main.go`) is a thin HTTP client. It doesn't do any indexing or embedding — it just talks to the Python server.

### Project Discovery

When you run `giddy`, it walks up from your current directory looking for:
1. A `.giddyanne.yaml` file (config mode)
2. A `.git/` directory (git-only mode)

This determines the project root and where to look for the server's storage directory.

### Server Discovery

The CLI looks for `.pid` files in the storage directory. The filename encodes the host and port (e.g., `-8000.pid` for default settings). It reads the PID, checks if the process is alive (`kill(pid, 0)`), and hits `/health` to verify.

### Auto-start

`giddy find` will start the server automatically if it's not running. It locates the Python venv relative to the `giddy` binary's installation directory, spawns the server in daemon mode, and polls `/health` every 500ms until it's ready.

### Command Matching

Commands can be abbreviated to any unambiguous prefix — `giddy f` for `find`, `giddy st` for `status`, etc.

## Storage Layout

Where files end up depends on whether you have a config file:

**Config mode** (`.giddyanne.yaml` exists):
```
<project>/.giddyanne/
├── all-MiniLM-L6-v2/     # Named after the embedding model
│   └── vectors.lance/    # LanceDB database
├── -8000.log             # Server log (host-port.log)
└── -8000.pid             # Server PID (host-port.pid)
```

**Git-only mode** (no config):
```
$TMPDIR/giddyanne/<project>-<hash>/
└── (same structure)
```

The hash is the first 8 characters of SHA256 of the absolute project path — deterministic, so the same project always gets the same storage location.

The database directory includes the model name. This means you can switch embedding models without corrupting existing indexes — each model gets its own database.

## Design Decisions

**Three embeddings per chunk** — path, content, and description each capture different signals. A query like "test helpers" benefits from the path embedding matching `tests/helpers.py`. A query like "authentication" benefits from the description embedding if the config says "auth and session management". Content embedding does the heavy lifting for everything else.

**Mtime-based skip** — comparing timestamps is cheap. Re-embedding is expensive. This single optimization makes re-indexing fast enough that you barely notice it on startup.

**Subprocess, not fork** — sentence-transformers loads PyTorch, which uses threads internally. Forking a threaded process is undefined behavior on most platforms. `Popen` spawns a clean new process.

**LanceDB** — it's an embedded database (no separate server process), stores data in Apache Arrow format, and supports both vector search and full-text search. One less thing to install and manage.

**Batch everything** — the embedding model call is the bottleneck. Batching 32 chunks into a single `encode()` call is dramatically faster than 32 individual calls. Same principle applies to database upserts.

**Debounced file watching** — editors often trigger multiple write events for a single save (write to temp file, rename, etc.). The 100ms debounce window catches all of these and fires once.
