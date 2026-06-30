"""Unit tests for scoped multi-tenant API keys."""
from __future__ import annotations

import pytest

from core.config.settings import settings
from core.security.keys import KeyRegistry, KeyPrincipal, ANONYMOUS


def test_single_key_backward_compatible():
    reg = KeyRegistry(single_key="topsecret", multi_key_spec="")
    p = reg.resolve("topsecret")
    assert p is not None and p.name == "default" and p.tenant == "*"
    assert reg.resolve("wrong") is None
    assert reg.resolve(None) is None


def test_multi_key_named_and_tenanted():
    reg = KeyRegistry(single_key="", multi_key_spec="svc-a:AAA:tenantA, svc-b:BBB")
    a = reg.resolve("AAA")
    b = reg.resolve("BBB")
    assert a.name == "svc-a" and a.tenant == "tenantA"
    assert b.name == "svc-b" and b.tenant == "*"      # no tenant => unrestricted
    assert reg.resolve("nope") is None


def test_newline_separated_spec():
    reg = KeyRegistry(single_key="", multi_key_spec="svc-a:AAA\nsvc-b:BBB:t2")
    assert reg.resolve("AAA").name == "svc-a"
    assert reg.resolve("BBB").tenant == "t2"


def test_auth_disabled_returns_anonymous():
    reg = KeyRegistry(single_key="", multi_key_spec="")
    assert reg.auth_disabled is True
    assert reg.resolve("anything") is ANONYMOUS
    assert reg.resolve(None) is ANONYMOUS


def test_principal_namespace_scoping():
    p = KeyPrincipal(name="svc", tenant="tenantA")
    assert p.may_access("tenantA") is True
    assert p.may_access("tenantA:user:1") is True
    assert p.may_access("tenantB") is False
    assert p.may_access("") is False
    # Unrestricted principal
    assert KeyPrincipal(name="root", tenant="*").may_access("anything") is True
