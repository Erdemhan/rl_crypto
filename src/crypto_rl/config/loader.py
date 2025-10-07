"""Config dosyalarini moduler ve tip-gvenli sekilde okuyan yardimci."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Sozlukleri derinlemesine birlestirir."""
    merged: Dict[str, Any] = dict(base or {})
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@dataclass
class ConfigBundle:
    """YAML dosyasindan okunan ham konfigurasyon bloklarini saklar."""

    meta: Dict[str, Any] = field(default_factory=dict)
    globals: Dict[str, Any] = field(default_factory=dict)
    profiles: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def materialize(self, profile: Optional[str] = None) -> Dict[str, Any]:
        """Profil secimine gore birlesik sozluk dondurur."""
        if profile is None:
            return dict(self.globals)
        if profile not in self.profiles:
            raise KeyError(f"Profil bulunamadi: {profile}")
        return _deep_merge(self.globals, self.profiles[profile])


class ConfigLoader:
    """Config okuyucu, eski Config sinifinin yeteneklerini moduler hale getirir."""

    def __init__(self, config_path: Optional[str] = None):
        self.path = self._resolve_path(config_path)
        self.bundle = self._load_bundle(self.path)

    # Genel API --------------------------------------------------------- #
    def get(self, key_path: str, default: Any = None, *, profile: Optional[str] = None) -> Any:
        """Nokta notasyonu ile ayar okur; ornek: ppo.learning_rate."""
        cursor: Any = self.bundle.materialize(profile)
        for segment in key_path.split("."):
            if not isinstance(cursor, dict) or segment not in cursor:
                return default
            cursor = cursor[segment]
        return cursor

    def resolved(self, profile: Optional[str] = None) -> Dict[str, Any]:
        """Tam birlesik sozlugu dondurur."""
        return self.bundle.materialize(profile)

    def profiles(self) -> Iterable[str]:
        """Calistirilabilir profilleri siralar."""
        order = self.bundle.meta.get("profiles_order")
        if isinstance(order, list):
            return [name for name in order if name in self.bundle.profiles]
        return self.bundle.profiles.keys()

    def meta(self) -> Dict[str, Any]:
        """Meta blokunu dondurur."""
        return dict(self.bundle.meta)

    # Yardimcilar ------------------------------------------------------- #
    @staticmethod
    def _resolve_path(path_hint: Optional[str]) -> Path:
        """Config dosyasini muhtemel yollarda arar."""
        candidates = [
            *(Path(p) for p in [path_hint] if p),
            Path("config.yaml"),
            Path("config/config.yaml"),
            Path("configs/config.yaml"),
        ]
        # Kullanicinin verdigi yol oncelikli, ardindan yaygin isimler deneniyor.
        for candidate in candidates:
            if candidate and candidate.exists():
                return candidate
        raise FileNotFoundError(
            "Config dosyasi bulunamadi. Denenen yollar: "
            + ", ".join(str(c) for c in candidates if c)
        )

    @staticmethod
    def _load_bundle(path: Path) -> ConfigBundle:
        """YAML icerigini ConfigBundle yapisina donusturur."""
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}

        if not isinstance(raw, dict):
            raise ValueError("Config dosyasinin kok ogesi bir sozluk olmali.")

        if {"meta", "globals", "profiles"} & raw.keys():
            meta = dict(raw.get("meta") or {})
            globals_cfg = dict(raw.get("globals") or {})
            profiles_cfg = {k: dict(v or {}) for k, v in (raw.get("profiles") or {}).items()}
        else:
            meta = {}
            globals_cfg = dict(raw)
            profiles_cfg = {}

        return ConfigBundle(meta=meta, globals=globals_cfg, profiles=profiles_cfg)


def load_config(config_path: Optional[str] = None) -> ConfigLoader:
    """Kisa yol yardimci fonksiyon."""
    return ConfigLoader(config_path=config_path)
