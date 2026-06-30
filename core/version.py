"""Single source of truth for the product version.

Imported by the API metadata/health responses and referenced by packaging and
docs so the version can never drift between them.
"""
from __future__ import annotations

__version__ = "0.3.0"

# Product branding. The public product is "stateful.ai"; "stateful.ai" is retained
# as the engine / Python package name for backward compatibility.
PRODUCT_NAME = "stateful.ai"
ENGINE_NAME = "stateful.ai"
