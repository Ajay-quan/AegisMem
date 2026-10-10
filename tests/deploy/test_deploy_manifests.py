"""Deployment invariants: rendered manifests and compose must keep the security model intact."""
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def k8s():
    if shutil.which("kubectl") is None:
        pytest.skip("kubectl not installed")
    out = subprocess.run(["kubectl", "kustomize", str(ROOT / "infra/k8s/overlays/production")],
                         capture_output=True, text=True, check=True).stdout
    return {d["kind"]: d for d in yaml.safe_load_all(out) if d}


def test_api_pods_are_hardened(k8s):
    pod = k8s["Deployment"]["spec"]["template"]["spec"]
    assert pod["securityContext"]["runAsNonRoot"] is True
    c = pod["containers"][0]
    assert c["securityContext"]["readOnlyRootFilesystem"] is True
    assert c["securityContext"]["allowPrivilegeEscalation"] is False
    assert c["securityContext"]["capabilities"]["drop"] == ["ALL"]
    assert c["readinessProbe"]["httpGet"]["path"] == "/ready"
    assert c["livenessProbe"]["httpGet"]["path"] == "/health"
    assert "limits" in c["resources"]


def test_production_config_is_safe(k8s):
    data = k8s["ConfigMap"]["data"]
    assert data["APP_ENV"] == "production"
    assert data["AUTH_ENABLED"] == "true" and data["REQUIRE_RLS"] == "true"
    assert data["RATE_LIMIT_BACKEND"] == "redis"  # in-process limits are wrong with >1 replica
    assert k8s["ServiceAccount"]["automountServiceAccountToken"] is False
    assert k8s["HorizontalPodAutoscaler"]["spec"]["minReplicas"] >= 2
    assert k8s["PodDisruptionBudget"]["spec"]["minAvailable"] >= 1


def test_migrations_use_owner_role_not_app_role(k8s):
    job = k8s["Job"]["spec"]["template"]["spec"]["containers"][0]
    assert job["command"] == ["alembic", "upgrade", "head"]
    assert any(e["name"] == "AEGIS_MIGRATION_URL" for e in job["env"])


def test_compose_runs_api_as_non_superuser_app_role():
    compose = yaml.safe_load((ROOT / "infra/compose/docker-compose.yml").read_text())
    api_env = compose["x-api-env"]
    assert api_env["POSTGRES_USER"] == "aegis_app" and api_env["AUTH_ENABLED"] == "true"
    assert compose["services"]["api"]["depends_on"]["migrate"]["condition"] == "service_completed_successfully"
    roles = (ROOT / "infra/compose/initdb/01-roles.sh").read_text()
    assert "aegis_app" in roles and "NOSUPERUSER NOBYPASSRLS" in roles
