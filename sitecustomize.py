"""Site-wide customizations loaded automatically by Python.

This file ensures **every** Python process in this virtual environment has
access to a recent SQLite runtime, which some third-party libraries (e.g.
ChromaDB) require.

Python executes ``import site`` on startup and, as documented, `site`
attempts to import ``sitecustomize`` if it exists on the import path.  By
placing this module at the project root (which is included in
``PYTHONPATH`` when our tooling runs) we guarantee that the monkey-patch
below is performed *before* any regular user code – including CLI tools
like ``chroma`` – touches ``sqlite3``.
"""

from __future__ import annotations

import sys


def _patch_sqlite() -> None:
    """Replace stdlib ``sqlite3`` with the modern `pysqlite3` build.

    If the wheel is not available we silently skip – consumers will get the
    system SQLite and may raise their own errors later.
    """

    try:
        import pysqlite3 as _pysqlite3  # noqa: WPS433 – late import OK

        # Any subsequent ``import sqlite3`` (direct or indirect) will now
        # receive the upgraded module.
        sys.modules["sqlite3"] = _pysqlite3  # type: ignore[assignment]
    except ModuleNotFoundError:
        # Fall back gracefully – the project’s requirements should install
        # the wheel, but we don't crash if it is missing.
        pass


# Perform the patch as early as possible.
_patch_sqlite()

# Clean-up internal names (optional but keeps ``help()`` tidy).
del _patch_sqlite