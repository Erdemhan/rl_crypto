"""Pipeline ortak yardımcıları."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


@dataclass
class RunPaths:
    """Her bir run için temel klasör yapısını tutar."""

    base: Path
    models: Path
    logs: Path
    results: Path
    validation: Path

    def ensure(self) -> None:
        """Klasörlerin var olduğundan emin ol."""
        for directory in [self.base, self.models, self.logs, self.results, self.validation]:
            directory.mkdir(parents=True, exist_ok=True)


class DictConfigAdapter:
    """Sözlük tabanlı ayarlara `.get` arabirimi kazandırır."""

    def __init__(self, data: Dict[str, Any]):
        self._data = data

    def get(self, key_path: str, default: Any = None) -> Any:
        cursor: Any = self._data
        for part in key_path.split("."):
            if not isinstance(cursor, dict) or part not in cursor:
                return default
            cursor = cursor[part]
        return cursor

    def set(self, key_path: str, value: Any) -> None:
        cursor = self._data
        parts = key_path.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value

    def data(self) -> Dict[str, Any]:
        return self._data


def set_seed(seed: int) -> None:
    """Deneylerin tekrar edilebilir olması için rastgelelik kaynaklarını sabitle."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_run_paths(meta: Dict[str, Any], profile: Optional[str], run_id: Optional[str]) -> RunPaths:
    """Meta bilgilerden çıktı klasörlerini üret."""
    root = Path(meta.get("profiles_dirname", "outputs"))
    suffix = profile or "default"
    run_name = f"{run_id}_{suffix}" if run_id else suffix
    base = root / run_name
    return RunPaths(
        base=base,
        models=base / "models",
        logs=base / "logs",
        results=base / "results",
        validation=base / "validation",
    )


def setup_logging(log_dir: Path, profile: Optional[str], prefix: str) -> Path:
    """Dosya ve konsol loglamasını hazırlar."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{profile}_{prefix}_{timestamp}.log" if profile else f"{prefix}_{timestamp}.log"
    log_path = log_dir / filename

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logging.getLogger().info("%s log dosyası: %s", prefix.capitalize(), log_path)
    return log_path
