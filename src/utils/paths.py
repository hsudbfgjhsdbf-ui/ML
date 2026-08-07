"""Repository path utilities used by every pipeline stage.

Keeping path resolution in one module prevents notebooks and model modules
from silently writing artifacts to different working directories.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ProjectPaths:
    """Absolute paths for the repository's declared artifact locations.

    Args:
        root: Repository root. It is resolved once at pipeline startup.
    Returns:
        A path object exposing stable locations for inputs and outputs.
    """

    root: Path

    @property
    def config(self) -> Path:
        """Return the configuration directory."""
        return self.root / "config"

    @property
    def raw_data(self) -> Path:
        """Return the raw data directory."""
        return self.root / "data" / "raw"

    @property
    def processed_data(self) -> Path:
        """Return the processed data directory."""
        return self.root / "data" / "processed"

    @property
    def interim_data(self) -> Path:
        """Return the interim data directory."""
        return self.root / "data" / "interim"

    @property
    def evaluation(self) -> Path:
        """Return the evaluation artifact directory."""
        return self.root / "evaluation"

    @property
    def documentation(self) -> Path:
        """Return the documentation directory."""
        return self.root / "documentation"

    @property
    def images(self) -> Path:
        """Return the visualization directory."""
        return self.root / "images"

    @property
    def models(self) -> Path:
        """Return the serialized model directory."""
        return self.root / "artifacts" / "models"

    @property
    def reports(self) -> Path:
        """Return the PDF report directory."""
        return self.root / "reports"

    @property
    def presentation(self) -> Path:
        """Return the presentation directory."""
        return self.root / "presentation"

    @property
    def workspace(self) -> Path:
        """Return a disposable scratch directory."""
        return self.root / "workspace"

    def ensure(self, extras: Iterable[Path] = ()) -> None:
        """Create declared artifact directories if they do not exist.

        Args:
            extras: Additional directories needed by a caller.
        Returns:
            None.
        Raises:
            OSError: If a directory cannot be created.
        """
        directories = [
            self.raw_data,
            self.processed_data,
            self.interim_data,
            self.evaluation,
            self.documentation,
            self.images / "eda",
            self.images / "models",
            self.images / "xai",
            self.images / "diagrams",
            self.models,
            self.reports / "assets",
            self.presentation / "assets",
            self.workspace,
            *extras,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


def find_repository_root(start: Path | None = None) -> Path:
    """Find the repository root by looking for configuration and source folders.

    Args:
        start: Optional starting path; the current working directory is used
            when omitted.
    Returns:
        An absolute repository root path.
    Raises:
        FileNotFoundError: If no repository marker can be found.
    """
    current = (start or Path.cwd()).resolve()
    candidates = [current, *current.parents]
    for candidate in candidates:
        if (candidate / "config").exists() and (candidate / "src").exists():
            return candidate
    raise FileNotFoundError("Could not find repository root: expected both config/ and src/ directories.")
