"""Allow running aldapy as a module: python -m aldapy"""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
