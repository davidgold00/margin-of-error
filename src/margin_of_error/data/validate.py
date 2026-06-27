"""Data validation entrypoint — `make data-check` runs this module.

Checks that all expected data files exist and, if found, validates them against
the pandera schemas and runs the data dictionary parser.

Exit codes:
    0 — all data that exists passed validation (missing files are a warning, not failure)
    1 — one or more existing files failed validation

Design: this module is intentionally standalone. It loads configs and data
directly so it can give clear, human-readable errors without depending on any
other src/ modules failing silently.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)


def _banner(text: str) -> None:
    width = 60
    print(f"\n{'─' * width}")
    print(f"  {text}")
    print(f"{'─' * width}")


def _ok(msg: str) -> None:
    print(f"  ✓  {msg}")


def _warn(msg: str) -> None:
    print(f"  ⚠  {msg}")


def _fail(msg: str) -> None:
    print(f"  ✗  {msg}")


def _check_file(path: Path, label: str) -> bool:
    """Return True if the file exists; print status either way."""
    if path.exists():
        size_kb = path.stat().st_size // 1024
        _ok(f"{label}: found ({size_kb} KB)")
        return True
    else:
        _warn(f"{label}: NOT FOUND at {path}")
        _warn("     → See data/README.md for download instructions")
        return False


def _validate_dataframe(df: Any, schema: Any, label: str) -> bool:
    """Run pandera validation and report results. Returns True on success."""
    try:
        schema.validate(df, lazy=True)
        _ok(f"{label}: schema validation passed ({len(df):,} rows)")
        return True
    except Exception as exc:
        _fail(f"{label}: schema validation FAILED")
        for line in str(exc).split("\n")[:10]:  # cap output at 10 lines
            print(f"       {line}")
        return False


def _check_data_dictionary(description_path: Path) -> bool:
    """Parse the data description file and report summary."""
    if not description_path.exists():
        _warn("data_description.txt: not found — skipping dictionary parse")
        return True  # not a hard failure

    try:
        from margin_of_error.data.dictionary import parse_description_file, summarize_dictionary

        dictionary = parse_description_file(description_path)
        summary = summarize_dictionary(dictionary)
        for line in summary.splitlines():
            _ok(line.strip())
        return True
    except Exception as exc:
        _fail(f"data_description.txt: parse failed — {exc}")
        return False


def _check_config_files() -> bool:
    """Verify that config YAML files exist and are loadable."""
    all_ok = True
    for config_path in ["config/economics.yaml", "config/model.yaml"]:
        p = Path(config_path)
        if not p.exists():
            _fail(f"{config_path}: MISSING — run from repo root")
            all_ok = False
            continue
        try:
            from margin_of_error.config import load_economics, load_model_config

            if "economics" in config_path:
                load_economics(p)
            else:
                load_model_config(p)
            _ok(f"{config_path}: loaded and validated")
        except Exception as exc:
            _fail(f"{config_path}: load failed — {exc}")
            all_ok = False
    return all_ok


def main() -> None:
    """Run all data validation checks. Exit 0 on success, 1 on failure."""
    print("\nMargin of Error — Data Validation Check")
    print("=" * 60)

    any_failure = False

    # ── 1. Config files ───────────────────────────────────────────────────────
    _banner("1 / 4  Config files")
    if not _check_config_files():
        any_failure = True

    # ── 2. Data file presence ─────────────────────────────────────────────────
    _banner("2 / 4  Data file presence")

    kaggle_train_path = Path("data/raw/kaggle/train.csv")
    kaggle_test_path = Path("data/raw/kaggle/test.csv")
    ames_full_path = Path("data/raw/ames/AmesHousing.csv")
    description_path = Path("data/raw/ames/data_description.txt")

    kaggle_train_exists = _check_file(kaggle_train_path, "Kaggle train.csv")
    kaggle_test_exists = _check_file(kaggle_test_path, "Kaggle test.csv")
    ames_full_exists = _check_file(ames_full_path, "Full Ames dataset")
    _check_file(description_path, "data_description.txt")

    found_count = sum([kaggle_train_exists, kaggle_test_exists, ames_full_exists])
    if found_count == 0:
        print()
        _warn("No data files found. Add data files per data/README.md and re-run.")
        print()
    else:
        _ok(f"{found_count}/3 primary data files found")

    # ── 3. Schema validation (only for files that exist) ─────────────────────
    _banner("3 / 4  Schema validation")

    if kaggle_train_exists or kaggle_test_exists or ames_full_exists:
        try:
            from margin_of_error.data.loaders import (
                load_ames_full,
                load_kaggle_test,
                load_kaggle_train,
            )
            from margin_of_error.data.schemas import (
                validate_ames_full,
                validate_kaggle_test,
                validate_kaggle_train,
            )

            if kaggle_train_exists:
                df_train = load_kaggle_train(kaggle_train_path)
                ncols = len(df_train.columns)
                _ok(f"Kaggle train: loaded ({len(df_train):,} rows, {ncols} columns)")
                try:
                    validate_kaggle_train(df_train)
                    _ok("Kaggle train: pandera schema passed")
                except Exception as exc:
                    _fail(f"Kaggle train: pandera FAILED — {str(exc)[:200]}")
                    any_failure = True

            if kaggle_test_exists:
                df_test = load_kaggle_test(kaggle_test_path)
                _ok(f"Kaggle test: loaded ({len(df_test):,} rows, {len(df_test.columns)} columns)")
                try:
                    validate_kaggle_test(df_test)
                    _ok("Kaggle test: pandera schema passed")
                except Exception as exc:
                    _fail(f"Kaggle test: pandera FAILED — {str(exc)[:200]}")
                    any_failure = True

            if ames_full_exists:
                df_ames = load_ames_full(ames_full_path)
                _ok(f"Full Ames: loaded ({len(df_ames):,} rows, {len(df_ames.columns)} columns)")
                try:
                    validate_ames_full(df_ames)
                    _ok("Full Ames: pandera schema passed")
                except Exception as exc:
                    _fail(f"Full Ames: pandera FAILED — {str(exc)[:200]}")
                    any_failure = True

        except ImportError as exc:
            _fail(f"Import error — is the package installed? (`make setup`) — {exc}")
            any_failure = True
    else:
        _warn("No data files to validate — skipping schema checks")

    # ── 4. Data dictionary ────────────────────────────────────────────────────
    _banner("4 / 4  Data dictionary")
    if not _check_data_dictionary(description_path):
        any_failure = True

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    if any_failure:
        _fail("Data check FAILED — see errors above")
        print()
        sys.exit(1)
    else:
        if found_count == 0:
            _warn("Data check passed (no data files present — add data to proceed)")
        else:
            _ok("Data check PASSED")
        print()
        sys.exit(0)


if __name__ == "__main__":
    main()
