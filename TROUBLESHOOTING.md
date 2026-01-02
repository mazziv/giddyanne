# Troubleshooting

## Common Issues

### "No results found"

1. **Check server status**: Run `giddy status` - is indexing complete?
2. **Rephrase your query**: Try describing what the code does differently
3. **Check paths**: Does your config include the directory you're searching?
4. **Try hybrid search**: If using `--semantic` or `--full-text`, try without flags

### "Config file not found"

You need a `.giddyanne.yaml` in your project root:

```bash
giddy init          # Shows a prompt to help you create one
```

Or create a minimal config:

```yaml
paths:
  - path: .
    description: My project
```

### Server won't start

**Port already in use:**
```bash
giddy down            # Stop any existing server
giddy up --port 9000  # Use a different port
```

**Check logs:**
```bash
giddy log             # Stream server logs
```

### Slow indexing

First index takes time (~45s for 750 files on M1 Pro). Subsequent startups are faster because:
- Unchanged files use cached embeddings
- Only modified files get re-indexed

If re-indexing is slow:
1. Check `giddy status` for progress
2. Consider excluding large directories via `ignore_patterns` in config
3. Check if files are being repeatedly modified

### High memory usage

The embedding model uses ~500MB RAM. If this is a problem:
- Use a smaller model: add `local_model: "paraphrase-MiniLM-L3-v2"` to config
- Stop the server when not in use: `giddy down`

### Search returns irrelevant results

1. **Add descriptions**: Better path descriptions improve ranking
2. **Be more specific**: Longer, more descriptive queries help
3. **Check file types**: Only [supported languages](README.md#supported-languages) are indexed

### Emacs/VSCode not connecting

1. Ensure the server is running: `giddy status`
2. Check you're in a directory with `.giddyanne.yaml`
3. For MCP: verify `giddy mcp` works from command line

## Internal Files

All runtime files live in `.giddyanne/` in your project root:

```
.giddyanne/
├── all-MiniLM-L6-v2/     # Named after your embedding model
│   ├── vectors.lance/    # Vector database (LanceDB)
│   └── mcp.log           # MCP server log
├── -8000.log             # HTTP server log (host-port.log)
└── -8000.pid             # Server PID file (host-port.pid)
```

### Log Files

**HTTP server log** (`.giddyanne/<host>-<port>.log`):
```bash
cat .giddyanne/-8000.log  # View directly
giddy log                  # Or stream live
```

Contains indexing progress, file watch events, and search requests.

**MCP server log** (`.giddyanne/<model>/mcp.log`):
```bash
cat .giddyanne/all-MiniLM-L6-v2/mcp.log
```

Contains MCP protocol messages and search requests from Claude Code.

### PID Files

The `.pid` file tracks the running server process:

```bash
cat .giddyanne/-8000.pid           # Shows PID
ps -p $(cat .giddyanne/-8000.pid)  # Verify process exists
```

If `giddy status` says the server is running but it's not responding, the PID file may be stale:

```bash
rm .giddyanne/-8000.pid
giddy up
```

### Database Files

The `vectors.lance/` directory is a LanceDB database containing your indexed embeddings.

**To force a re-index** (keeps logs):
```bash
giddy drop
```

**To reset everything** (removes all .giddyanne data):
```bash
giddy clean        # prompts for confirmation
giddy clean --force  # no confirmation
```
