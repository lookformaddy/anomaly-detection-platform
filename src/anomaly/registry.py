"""Versioned detector registry with rollback. Deploying a detector is a
release: versions are immutable once registered, exactly one version is
active per detector name, and rollback is a pointer move, not a redeploy."""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DetectorVersion:
    name: str
    version: int
    kind: str
    threshold: float
    params: dict
    active: bool = False


@dataclass
class DetectorRegistry:
    _versions: dict[str, list[DetectorVersion]] = field(default_factory=dict, repr=False)

    def register(self, name: str, kind: str, threshold: float, params: dict | None = None) -> DetectorVersion:
        versions = self._versions.setdefault(name, [])
        version = DetectorVersion(
            name=name, version=len(versions) + 1, kind=kind,
            threshold=threshold, params=params or {}, active=not versions,
        )
        versions.append(version)
        return version

    def activate(self, name: str, version: int) -> DetectorVersion:
        versions = self._versions.get(name, [])
        target = next((v for v in versions if v.version == version), None)
        if target is None:
            raise KeyError(f"no version {version} for detector {name!r}")
        self._versions[name] = [
            DetectorVersion(**{**v.__dict__, "active": v.version == version}) for v in versions
        ]
        return next(v for v in self._versions[name] if v.active)

    def rollback(self, name: str) -> DetectorVersion:
        """Activate the previous version. Fails loudly if there is none --
        rolling back to nothing is a deployment decision, not a default."""
        versions = self._versions.get(name, [])
        active = next((v for v in versions if v.active), None)
        if active is None or active.version <= 1:
            raise ValueError(f"nothing to roll back to for {name!r}")
        return self.activate(name, active.version - 1)

    def active(self, name: str) -> DetectorVersion | None:
        return next((v for v in self._versions.get(name, []) if v.active), None)

    def history(self, name: str) -> list[DetectorVersion]:
        return list(self._versions.get(name, []))

    def to_json(self) -> str:
        return json.dumps(
            {name: [v.__dict__ for v in vs] for name, vs in self._versions.items()},
            indent=2, sort_keys=True,
        )

    @classmethod
    def from_json(cls, payload: str) -> "DetectorRegistry":
        data = json.loads(payload)
        reg = cls()
        for name, vs in data.items():
            reg._versions[name] = [DetectorVersion(**v) for v in vs]
        return reg
