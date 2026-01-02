# Contributing

Contributions welcome!

## Development Setup

```bash
git clone https://github.com/mazziv/giddyanne.git
cd giddyanne

# Build everything
make

# Install dev dependencies
.venv/bin/pip install -e ".[dev]"
```

## Running Tests

```bash
# Linter
.venv/bin/ruff check .

# Tests
.venv/bin/pytest
```

## Project Structure

```
giddyanne/
├── cmd/giddy/main.go    # Go CLI
├── main.py              # Python HTTP server
├── mcp_main.py          # Python MCP server
├── emacs/               # Emacs package
└── vscode/              # VSCode extension
```

## Help Wanted

Here's what would help most:

### Windows Support

The Python server should work, but needs testing and documentation:
- Test installation and server on Windows
- Document any needed changes (paths, process management)
- Update Makefile or add Windows-specific install script

### Vim/Neovim Plugin

A plugin that calls the CLI and populates quickfix:
- `giddy find` integration with fzf or telescope
- Results to quickfix list
- Basic commands: up, down, status

Open an issue to discuss or submit a PR.
