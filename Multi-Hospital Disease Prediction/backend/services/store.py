"""Read-only access to results/*.json (the pipeline output). Missing files -> ResultsMissing (HTTP 503)."""
from __future__ import annotations

import json
from pathlib import Path

from ml.jsonutil import clean_json

REQUIRED = ["comparison.json", "client_summary.json", "privacy_curve.json", "attack.json",
            "communication.json", "run_meta.json", "preprocess_report.json"]


class ResultsMissing(Exception):
    def __init__(self, name: str):
        self.name = name
        super().__init__(f"Results file '{name}' not found. Run `python -m ml.run_all` first.")


class Store:
    def __init__(self, results_dir: Path):
        self.dir = Path(results_dir)
        self._cache: dict[str, dict] = {}

    def exists(self, name: str) -> bool:
        return (self.dir / name).is_file()

    def all_loaded(self) -> bool:
        return all(self.exists(n) for n in REQUIRED)

    def get(self, name: str):
        if name not in self._cache:
            p = self.dir / name
            if not p.is_file():
                raise ResultsMissing(name)
            self._cache[name] = clean_json(json.loads(p.read_text(encoding="utf-8")))
        return self._cache[name]

    def scenarios(self) -> list[str]:
        out = [p.stem.removeprefix("rounds_") for p in sorted(self.dir.glob("rounds_*.json"))]
        return out

    def rocs(self) -> list[str]:
        return [p.stem.removeprefix("roc_") for p in sorted(self.dir.glob("roc_*.json"))]
