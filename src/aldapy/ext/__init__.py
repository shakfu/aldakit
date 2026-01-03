"""Vendored external packages."""

import sys
from pathlib import Path

# Add ext directory to sys.path so vendored packages can import each other
_ext_dir = Path(__file__).parent
if str(_ext_dir) not in sys.path:
    sys.path.insert(0, str(_ext_dir))
