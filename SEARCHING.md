# Searching

How to get the most out of giddyanne's semantic search.

## How It Works

Giddyanne uses embedding models to understand the *meaning* of code, not just match keywords:

- **"auth logic"** finds `login.py`, `session_manager.py`, `permissions.go`
- **"error handling"** finds try/catch blocks, error types, recovery code
- **"database queries"** finds SQL, ORM calls, connection pooling

The search combines two approaches:
- **Semantic search**: Finds code with similar meaning to your query
- **Full-text search**: Finds exact matches for specific terms

By default, results use both (hybrid search). Use `--semantic` or `--full-text` to use only one.

## Writing Good Queries

### What Works Well

**Conceptual queries** - describe what the code does:
```bash
giddy find "user authentication"
giddy find "parse command line arguments"
giddy find "send email notifications"
```

**Problem-oriented queries** - describe what you're trying to solve:
```bash
giddy find "handle network timeouts"
giddy find "validate user input"
giddy find "cache expensive computations"
```

**Behavior descriptions** - describe what happens:
```bash
giddy find "retry failed requests"
giddy find "log errors to file"
giddy find "convert dates to timestamps"
```

### What Works Less Well

**Variable names** - use grep instead:
```bash
# Less effective:
giddy find "userAuthToken"

# Better:
grep -r "userAuthToken" src/
```

**Very short queries** - add context:
```bash
# Too vague:
giddy find "config"

# Better:
giddy find "load configuration from yaml file"
```

## Tips

1. **Use natural language** - write queries like you'd describe the code to someone
2. **Be specific** - "database connection pooling" beats "database stuff"
3. **Include verbs** - "validate", "parse", "send", "handle" help semantic matching
4. **Add context** - "http error handling" is better than just "errors"

## Search Modes

```bash
giddy find "auth"              # Hybrid (default) - combines both approaches
giddy find "auth" --semantic   # Semantic only - meaning-based matching
giddy find "auth" --full-text  # Full-text only - keyword matching
```

When to use each:
- **Hybrid**: Best for most queries
- **Semantic**: When you're describing functionality conceptually
- **Full-text**: When you know specific terms that should appear in the code
