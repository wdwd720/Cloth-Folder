"""Enable ``python -m terafold ...`` as an alias for the ``terafold`` CLI."""

from __future__ import annotations

from terafold.cli import app

if __name__ == "__main__":
    app()
