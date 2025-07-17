"""
Flash Report Generator package.

This package loads a modern SQLite runtime (via the ``pysqlite3`` wheel)
*before* anything else touches ``sqlite3``.  ChromaDB requires
``sqlite3 >= 3.35`` and many Linux distributions ship an older build.

If ``pysqlite3`` is available we monkey-patch the standard library
module slot so that every subsequent ``import sqlite3`` (including those
that take place deep inside ChromaDB) receives the newer implementation.

This happens at ``import src`` time which means it will execute once
and **very** early – well before the rest of our application (or
third-party libraries) have a chance to import ``sqlite3`` on their own.
"""

# ---------------------------------------------------------------------------
# Ensure ChromaDB sees a recent SQLite version
# ---------------------------------------------------------------------------

try:
    # The binary wheel bundles a recent SQLite (≥3.38).  We import it and
    # then overwrite the ``sqlite3`` entry in ``sys.modules`` so that the
    # standard library name resolves to this upgraded build everywhere.
    import pysqlite3 as _pysqlite3  # noqa: WPS433  (third-party import alias)
    import sys

    sys.modules["sqlite3"] = _pysqlite3  # type: ignore[assignment]
except ModuleNotFoundError:
    # ``pysqlite3-binary`` isn't installed – fall back to the system
    # sqlite3.  ChromaDB will raise a clear error later if it is too old.
    pass


# Package metadata
__version__ = "0.1.0"
__author__ = "Your Name"
__license__ = "MIT" 