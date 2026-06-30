"""Scoped, revocable, multi-tenant API keys.

The original auth was a single global ``STATEFUL_AI_API_KEY`` — fine for a demo,
but production multi-tenant deployments need to issue and revoke a *named* key
per consumer and attribute every request to a tenant. This module adds that
without breaking backward compatibility:

* ``STATEFUL_AI_API_KEY`` (single key) still works — it becomes a principal named
  ``default`` with tenant ``*`` (all namespaces).
* ``STATEFUL_AI_API_KEYS`` holds multiple keys, one per line or comma-separated,
  each ``name:secret`` or ``name:secret:tenant``. Revoking a consumer is just
  removing its line.

Matching is constant-time across all configured keys (``hmac.compare_digest``)
to avoid timing side-channels. Resolution returns a :class:`KeyPrincipal` that
downstream code attaches to the request for attribution and (optional) tenant
scoping.
"""
from __future__ import annotations

import hmac
from dataclasses import dataclass

from core.config.settings import settings


@dataclass(frozen=True)
class KeyPrincipal:
    """The identity behind a validated request."""

    name: str
    tenant: str = "*"          # "*" => unrestricted (all namespaces)
    anonymous: bool = False    # True when auth is disabled entirely

    def may_access(self, namespace: str) -> bool:
        """Whether this principal may touch the given namespace."""
        if self.tenant == "*" or self.anonymous:
            return True
        if not namespace:
            return False
        return namespace == self.tenant or namespace.startswith(f"{self.tenant}:")


ANONYMOUS = KeyPrincipal(name="anonymous", tenant="*", anonymous=True)


class KeyRegistry:
    """Holds configured keys and resolves a presented secret to a principal."""

    def __init__(self, single_key: str, multi_key_spec: str) -> None:
        # secret -> KeyPrincipal
        self._keys: dict[str, KeyPrincipal] = {}
        if single_key:
            self._keys[single_key] = KeyPrincipal(name="default", tenant="*")
        for entry in self._split(multi_key_spec):
            parts = entry.split(":")
            if len(parts) < 2:
                continue
            name, secret = parts[0].strip(), parts[1].strip()
            tenant = parts[2].strip() if len(parts) >= 3 and parts[2].strip() else "*"
            if name and secret:
                self._keys[secret] = KeyPrincipal(name=name, tenant=tenant)

    @staticmethod
    def _split(spec: str) -> list[str]:
        if not spec:
            return []
        normalized = spec.replace("\n", ",")
        return [e for e in (p.strip() for p in normalized.split(",")) if e]

    @property
    def auth_disabled(self) -> bool:
        return not self._keys

    def resolve(self, presented: str | None) -> KeyPrincipal | None:
        """Constant-time resolve a presented secret to its principal.

        Returns ``ANONYMOUS`` when auth is disabled, ``None`` when the key is
        missing or invalid, otherwise the matching principal.
        """
        if self.auth_disabled:
            return ANONYMOUS
        if not presented:
            return None
        presented_bytes = presented.encode()
        match: KeyPrincipal | None = None
        # Compare against every key (no early exit) to keep timing uniform.
        for secret, principal in self._keys.items():
            if hmac.compare_digest(presented_bytes, secret.encode()):
                match = principal
        return match

    def principals(self) -> list[str]:
        return sorted({p.name for p in self._keys.values()})


def get_key_registry() -> KeyRegistry:
    """Build the registry from *current* settings on each call.

    Auth config (single key + scoped keys) is read live so changes to settings
    (e.g. in tests or hot reload) take effect immediately, matching the prior
    live-read auth behavior. Building over a handful of keys is negligible.
    """
    return KeyRegistry(
        single_key=settings.api_key,
        multi_key_spec=getattr(settings, "api_keys", ""),
    )


def reset_key_registry() -> None:
    """No-op kept for API compatibility (the registry is now built per call)."""
    return None
