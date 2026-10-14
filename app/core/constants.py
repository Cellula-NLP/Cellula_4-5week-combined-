"""Shared constants that don't belong to a specific layer."""
from __future__ import annotations

SUPPORTED_LANGUAGES: frozenset[str] = frozenset({
    "python", "javascript", "typescript", "java", "go",
    "rust", "c", "cpp", "csharp", "ruby", "php", "swift", "kotlin",
})

ALLOWED_UPLOAD_EXTENSIONS: frozenset[str] = frozenset({
    ".pdf", ".txt", ".md", ".py", ".js", ".ts", ".java",
    ".go", ".rs", ".c", ".cpp", ".rb", ".php",
})

EXECUTABLE_LANGUAGES: frozenset[str] = frozenset({"python", "javascript", "bash"})

DEFAULT_RUNNER = "subprocess"