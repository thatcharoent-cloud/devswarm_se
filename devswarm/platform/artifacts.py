"""
devswarm.platform.artifacts — Run artifact store.

Manages outputs, reports, diffs, gate logs, and manifests under:
.swarm/runs/<run_id>/artifacts/
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from devswarm.platform.events import redact_secrets

logger = logging.getLogger(__name__)


@dataclass
class ArtifactMeta:
    """Metadata recorded for an artifact file."""
    name: str
    relative_path: str
    content_type: str
    size_bytes: int
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class ArtifactStore:
    """Stores and catalogues generated artifacts for a single run."""

    def __init__(self, run_dir: Path | str) -> None:
        self.artifacts_dir = Path(run_dir) / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.artifacts_dir / "index.json"

    def write_artifact(
        self,
        filename: str,
        content: str | bytes,
        content_type: str = "text/plain",
    ) -> Path:
        """Write content to an artifact file and update the index."""
        target = self.artifacts_dir / filename
        target.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(content, str):
            clean = redact_secrets(content)
            target.write_text(clean, encoding="utf-8")
            size = len(clean.encode("utf-8"))
        else:
            target.write_bytes(content)
            size = len(content)

        meta = ArtifactMeta(
            name=filename,
            relative_path=str(target.relative_to(self.artifacts_dir)),
            content_type=content_type,
            size_bytes=size,
        )
        self._record_index(meta)
        return target

    def _record_index(self, meta: ArtifactMeta) -> None:
        index: list[dict[str, Any]] = []
        if self.manifest_path.exists():
            try:
                index = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except Exception:
                index = []

        # Replace or append
        index = [item for item in index if item.get("name") != meta.name]
        index.append({
            "name": meta.name,
            "relative_path": meta.relative_path,
            "content_type": meta.content_type,
            "size_bytes": meta.size_bytes,
            "created_at": meta.created_at,
        })
        self.manifest_path.write_text(json.dumps(index, indent=2), encoding="utf-8")

    def list_artifacts(self) -> list[dict[str, Any]]:
        if not self.manifest_path.exists():
            return []
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return []

