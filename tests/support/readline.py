"""Test-only shim for environments where importing macOS readline segfaults.

Pytest imports ``readline`` during capture startup. This project runs correctly
without interactive line editing, so the Makefile prepends this directory to
``PYTHONPATH`` for tests only.
"""


def set_completer(*args: object, **kwargs: object) -> None:
    return None


def get_completer() -> None:
    return None


def parse_and_bind(*args: object, **kwargs: object) -> None:
    return None


def read_init_file(*args: object, **kwargs: object) -> None:
    return None


def set_history_length(*args: object, **kwargs: object) -> None:
    return None
