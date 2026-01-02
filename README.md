# Giddyanne

Semantic codebase search. Indexes your files with embeddings and lets you find code by meaning, not just keywords.

**v1.1.1** · [Changelog](CHANGELOG.md)

- **Semantic search**: Find code by meaning ("auth logic" finds `login.py`)
- **Real-time indexing**: Watches for file changes and re-indexes automatically
- **Local-first**: Runs sentence-transformers locally, no API keys needed
- **Private**: Your code stays on your machine
- **Fast**: Pre-indexed embeddings, search in milliseconds
- **Useful**: Integrations for Emacs, VSCode, MCP, HTTP, & CLI

## Quickstart

```bash
git clone --depth 1 --branch v1.1.1 https://github.com/mazziv/giddyanne.git
cd giddyanne && make install
```

Then in your project:

```bash
giddyanne · main ⟩ giddy up
> Starting server...
> Server started on port 8000

giddyanne · main ⟩ giddy health
> Indexed files: 28
> Total chunks:  288
> Index size:    3.85 MB

giddyanne · main ⟩ giddy find "formatting functions"
> cmd/giddy/main.go:889-906 (0.31)
>   func formatBytes(bytes int64) string { ... }

giddyanne · main ⟩ giddy down
> Server stopped
```

Requires a `.giddyanne.yaml` config file. See [CONFIG.md](CONFIG.md) or run `giddy init`.

## Supported Languages

Only files with supported extensions are indexed. Code is split at natural boundaries (functions, classes) for better results.

**Programming:** C, C++, C#, Dart, Elixir, Go, Java, JavaScript, Kotlin, Lua, Objective-C, PHP, Python, R, Ruby, Rust, Scala, Shell, SQL, Swift, TypeScript, Zig

**Markup & config:** CSS, HTML, JSON, Markdown, TOML, YAML

Files matching `.gitignore` patterns are automatically excluded.

## Documentation

| Doc | Description |
|-----|-------------|
| [INSTALL.md](INSTALL.md) | Installation options and prerequisites |
| [CONFIG.md](CONFIG.md) | Configuration file reference |
| [CLI.md](CLI.md) | Command and flag reference |
| [SEARCHING.md](SEARCHING.md) | How to write effective queries |
| [INTEGRATIONS.md](INTEGRATIONS.md) | MCP, Emacs, VSCode, HTTP API |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Common issues and solutions |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Development setup and help wanted |
