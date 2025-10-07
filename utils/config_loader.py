"""Eski Config arabirimini yeni loader'a yonlendiren sarici."""

from __future__ import annotations

from typing import Any, Dict, Optional

from crypto_rl.config.loader import ConfigLoader


class Config:
    """Geri uyumluluk icin ince sarici."""

    def __init__(self, config_path: Optional[str] = None, profile: Optional[str] = None):
        self._loader = ConfigLoader(config_path=config_path)
        self._selected_profile = profile

    def get(self, key_path: str, default: Any = None, profile: Optional[str] = None) -> Any:
        target_profile = profile if profile is not None else self._selected_profile
        return self._loader.get(key_path, default=default, profile=target_profile)

    def all(self, profile: Optional[str] = None) -> Dict[str, Any]:
        target_profile = profile if profile is not None else self._selected_profile
        return self._loader.resolved(target_profile)

    def profiles(self):
        return list(self._loader.profiles())

    def meta(self) -> Dict[str, Any]:
        return self._loader.meta()
