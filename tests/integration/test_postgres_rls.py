"""Postgres RLS enforcement (runs in CI where AEGIS_TEST_PG_URL is set)."""
import os

import pytest

from adapters.relational_store.postgres_store import PostgresStore
from adapters.tenancy import TenantScopedMemoryStore
from core.exceptions import MemoryNotFoundError
from core.schemas.memory import MemoryItem
from core.tenancy_context import system_context

PG_URL = os.environ.get("AEGIS_TEST_PG_URL")
pytestmark = pytest.mark.skipif(not PG_URL, reason="needs Postgres (set in CI)")

PRED = ("tenant_id = current_setting('app.tenant_id', true) "
        "OR current_setting('app.rls_bypass', true) = 'on'")


@pytest.fixture
async def db():
    from adapters.relational_store import control_models  # noqa: F401
    from adapters.relational_store.models import Base
    store = PostgresStore(PG_URL)
    async with store._engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
        for stmt in ("ALTER TABLE memories ENABLE ROW LEVEL SECURITY",
                     "ALTER TABLE memories FORCE ROW LEVEL SECURITY",
                     f"CREATE POLICY tenant_isolation ON memories USING ({PRED}) WITH CHECK ({PRED})"):
            await conn.exec_driver_sql(stmt)
    yield store
    async with store._engine.begin() as conn:
        await conn.exec_driver_sql("DROP POLICY IF EXISTS tenant_isolation ON memories")
        await conn.exec_driver_sql("ALTER TABLE memories NO FORCE ROW LEVEL SECURITY")
        await conn.exec_driver_sql("ALTER TABLE memories DISABLE ROW LEVEL SECURITY")
    await store.close()


async def test_database_refuses_cross_tenant_rows(db):
    a = TenantScopedMemoryStore(db, "acme")
    m = await a.save_memory(MemoryItem(namespace="n", user_id="u", content="acme secret"))
    assert (await a.get_memory(m.memory_id)).content == "acme secret"

    # Raw access with no tenant context sees nothing (fail closed) ...
    with pytest.raises(MemoryNotFoundError):
        await db.get_memory(m.memory_id)
    assert await db.list_memories(user_id="u") == []
    # ... another tenant sees nothing, even bypassing the app-level wrapper ...
    from core.tenancy_context import tenant_context
    with tenant_context("globex"):
        with pytest.raises(MemoryNotFoundError):
            await db.get_memory(m.memory_id)
    # ... and only an explicit system context can see across tenants.
    with system_context():
        assert (await db.get_memory(m.memory_id)).tenant_id == "acme"


async def test_database_refuses_writes_into_another_tenant(db):
    from core.tenancy_context import tenant_context
    with tenant_context("globex"):
        with pytest.raises(Exception):  # RLS WITH CHECK violation
            await db.save_memory(MemoryItem(tenant_id="acme", namespace="n", user_id="u", content="forged"))
