"""Reproducible source digest for run and queue manifests.

The digest covers implementation, configs, dependency files and .gitignore,
not the README, artifacts, virtual environments, caches or Git internals. The
README is narrative evidence that records digest values, so including it would
make the digest circular. Per-file hashes make the result reproducible from the
recorded file list, and Git HEAD/dirty state is recorded separately so
historical evidence can be traced to a revision.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT

INCLUDED_SUFFIXES = {".py", ".yaml", ".yml", ".md", ".txt"}
EXCLUDED_DIR_NAMES = {
    ".git",
    ".venv",
    ".tools",
    "artifacts",
    "__pycache__",
    ".pytest_cache",
}
INCLUDED_TOP_LEVEL_FILES = {"requirements.txt", "requirements-lock.txt", ".gitignore"}
DOCUMENTATION_EXCLUDED = ("README.md",)


def _included_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for directory in (root / "rl_a2", root / "scripts", root / "configs"):
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            if not path.is_file() or path.suffix not in INCLUDED_SUFFIXES:
                continue
            if any(part in EXCLUDED_DIR_NAMES for part in path.relative_to(root).parts):
                continue
            files.append(path)
    for name in sorted(INCLUDED_TOP_LEVEL_FILES):
        path = root / name
        if path.is_file():
            files.append(path)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def source_digest(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Return a canonical SHA-256 plus per-file hashes and Git state."""
    root = Path(root).resolve()
    files = _included_files(root)
    per_file: dict[str, str] = {}
    combined = hashlib.sha256()
    for path in files:
        rel = path.relative_to(root).as_posix()
        payload = path.read_bytes()
        file_hash = hashlib.sha256(payload).hexdigest()
        per_file[rel] = file_hash
        combined.update(rel.encode("utf-8"))
        combined.update(b"\0")
        combined.update(str(len(payload)).encode("ascii"))
        combined.update(b"\0")
        combined.update(payload)
        combined.update(b"\0")

    def git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            return None

    head = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    return {
        "algorithm": "sha256 over sorted relative path, byte length and bytes of each source file",
        "digest": combined.hexdigest(),
        "file_count": len(files),
        "files": per_file,
        "git_head": head,
        "git_status_dirty": bool(status),
        "git_status_lines": len(status.splitlines()) if status else 0,
        "excluded": sorted(EXCLUDED_DIR_NAMES),
        "documentation_excluded": list(DOCUMENTATION_EXCLUDED),
    }
