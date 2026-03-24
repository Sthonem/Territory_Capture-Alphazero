"""Small replay-buffer utilities for self-play training data."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List


@dataclass
class ReplayBuffer:
    """A simple fixed-size buffer that stores recent self-play samples."""

    max_samples: int = 10_000
    records: List[dict] = field(default_factory=list)

    def extend(self, new_records: Iterable[dict]) -> None:
        """Append new samples and trim older entries when over capacity."""

        self.records.extend(new_records)
        overflow = len(self.records) - self.max_samples
        if overflow > 0:
            self.records = self.records[overflow:]

    def load_json(self, path: str | Path) -> None:
        """Append records from a JSON export if the file exists."""

        source = Path(path)
        if not source.exists():
            return
        self.extend(json.loads(source.read_text(encoding="utf-8")))

    def save_json(self, path: str | Path) -> Path:
        """Write the current replay buffer contents to JSON."""

        destination = Path(path).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.records), encoding="utf-8")
        return destination
