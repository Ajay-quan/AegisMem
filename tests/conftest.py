"""Root test configuration."""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest_plugins = ["tests.fixtures.conftest"]


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
async def _isolated_control_plane(tmp_path):
    """Every test gets its own durable control plane (SQLite file) - never the dev DB."""
    from adapters.relational_store.control_store import ControlStore
    from apps.api import dependencies as deps
    from apps.api.main import app
    from services.learning_runtime import LearningRuntime

    store = ControlStore(f"sqlite+aiosqlite:///{tmp_path}/control.db")
    await store.create_schema()
    runtime = LearningRuntime(store, hash_secret="test-secret")
    app.dependency_overrides[deps.get_control_store] = lambda: store
    app.dependency_overrides[deps.get_runtime] = lambda: runtime
    yield store
    app.dependency_overrides.pop(deps.get_control_store, None)
    app.dependency_overrides.pop(deps.get_runtime, None)
    await store.close()
