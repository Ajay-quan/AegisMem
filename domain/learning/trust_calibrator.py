"""Outcome-calibrated trust: the memory system learns which sources to believe.

Static trust tables ("tool output = 0.5") are wrong for most real sources. A CRM sync is
nearly always right, while a web-search tool hitting forum posts often is not. We learn this
from outcomes the product already produces:

    quarantine_released   a reviewer approved held content        strong positive
    quarantine_rejected   a reviewer rejected held content        strong negative
    corrected             the user/agent corrected a memory       negative
    confirmed             the user explicitly confirmed a memory  positive
    used                  a retrieved memory was used by the agent weak positive

Each source (``user_message``, ``tool_output``...) and each origin within it (``tool:web_search``,
``domain:reddit.com``) has a Beta posterior. Origins shrink toward their source, so a brand-new
origin starts at the source's learned trust, not a hard-coded constant.

Trust is learned **per deployment, across users**: one rejection of a poisoned domain protects
every user. The guard applies learned trust *retroactively*, so memories already stored from a
newly distrusted origin stop being served immediately, without a rewrite or re-index.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from domain.integrity.provenance import DEFAULT_TRUST, SOURCE_TRUST

# (alpha increment, beta increment) per outcome.
EVENT_WEIGHTS: dict[str, tuple[float, float]] = {
    "quarantine_released": (2.0, 0.0),
    "quarantine_rejected": (0.0, 4.0),
    "corrected": (0.0, 2.0),
    "confirmed": (1.0, 0.0),
    "used": (0.25, 0.0),
}


@dataclass
class BetaStat:
    alpha: float
    beta: float
    events: int = 0
    updated_at: datetime | None = None

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)


def _key(source: str, origin: str = "") -> str:
    return f"{source}|{origin}" if origin else source


class InMemoryTrustStore:
    def __init__(self) -> None:
        self._stats: dict[str, BetaStat] = {}

    async def get(self, key: str) -> BetaStat | None:
        return self._stats.get(key)

    async def put(self, key: str, stat: BetaStat) -> None:
        self._stats[key] = stat

    async def items(self) -> list[tuple[str, BetaStat]]:
        return list(self._stats.items())


class TrustCalibrator:
    def __init__(self, store: InMemoryTrustStore | None = None, prior_strength: float = 10.0) -> None:
        self._store = store or InMemoryTrustStore()
        self.k = prior_strength

    async def _source_stat(self, source: str) -> BetaStat:
        stat = await self._store.get(_key(source))
        if stat is None:
            p = SOURCE_TRUST.get(source, DEFAULT_TRUST)
            stat = BetaStat(alpha=p * self.k, beta=(1 - p) * self.k)
        return stat

    async def _origin_stat(self, source: str, origin: str) -> BetaStat:
        stat = await self._store.get(_key(source, origin))
        if stat is None:
            p = (await self._source_stat(source)).mean
            stat = BetaStat(alpha=p * self.k, beta=(1 - p) * self.k)
        return stat

    async def trust(self, source: str, origin: str = "") -> float:
        stat = await (self._origin_stat(source, origin) if origin else self._source_stat(source))
        return stat.mean

    async def record(self, source: str, origin: str, event: str, weight: float = 1.0) -> float:
        """Apply an outcome. Updates the origin and (more weakly) its parent source."""
        if event not in EVENT_WEIGHTS:
            raise ValueError(f"unknown trust event: {event}")
        da, db = EVENT_WEIGHTS[event]
        now = datetime.now(timezone.utc)

        # Source-level learns at half rate when an origin is known: one bad domain
        # should not condemn every tool output.
        src_w = weight * (0.5 if origin else 1.0)
        src = await self._source_stat(source)
        src.alpha += da * src_w
        src.beta += db * src_w
        src.events += 1
        src.updated_at = now
        await self._store.put(_key(source), src)

        if not origin:
            return src.mean
        org = await self._origin_stat(source, origin)
        org.alpha += da * weight
        org.beta += db * weight
        org.events += 1
        org.updated_at = now
        await self._store.put(_key(source, origin), org)
        return org.mean

    async def snapshot(self) -> list[dict[str, Any]]:
        rows = []
        for key, s in await self._store.items():
            source, _, origin = key.partition("|")
            prior = SOURCE_TRUST.get(source, DEFAULT_TRUST)
            rows.append({
                "source": source,
                "origin": origin,
                "trust": round(s.mean, 4),
                "default_trust": prior,
                "events": s.events,
                "updated_at": s.updated_at.isoformat() if s.updated_at else None,
            })
        rows.sort(key=lambda r: r["trust"])
        return rows


def memory_source_origin(metadata: dict[str, Any], fallback_source: Any = "") -> tuple[str, str]:
    prov = metadata.get("provenance") or {}
    source = prov.get("source") or (fallback_source.value if hasattr(fallback_source, "value") else str(fallback_source))
    return source, prov.get("origin", "")
