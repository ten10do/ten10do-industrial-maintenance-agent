"""Shortcut entry point: ``python -m app.database``."""

from app.database.init_db import main

if __name__ == "__main__":
    raise SystemExit(main())
