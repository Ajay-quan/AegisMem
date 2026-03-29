"""Root test configuration."""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest_plugins = ["tests.fixtures.conftest"]
