"""
logger.py
=========
Structured, coloured (ANSI) console logging for the Khazan Material Importer.

All output goes to sys.stdout so it appears in Blender's System Console
(Window > Toggle System Console on Windows) and in the terminal.
"""

from __future__ import annotations

import sys
from typing import List

from .node_builder import BuildResult


# ---------------------------------------------------------------------------
# ANSI colour helpers
# ---------------------------------------------------------------------------
_RESET = "\033[0m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_CYAN = "\033[96m"
_BOLD = "\033[1m"
_DIM = "\033[2m"


def _c(text: str, colour: str) -> str:
    """Wrap text in ANSI colour codes."""
    return f"{colour}{text}{_RESET}"


# ---------------------------------------------------------------------------
# Single-material report
# ---------------------------------------------------------------------------
def log_material_result(result: BuildResult, verbose: bool = False) -> None:
    """Print a structured report for one material to stdout."""
    sep = _c("-" * 56, _DIM)
    print(sep)
    print(_c(f"Material:  {result.material_name}", _BOLD))

    if result.was_dry_run:
        print(_c("  [DRY RUN – no changes made]", _YELLOW))

    if result.json_matched:
        print(_c(f"  Matched:  {result.json_matched}.json", _GREEN))
    else:
        print(_c("  Matched:  (none – JSON not found)", _RED))

    for tex in result.textures_loaded:
        print(_c(f"  Loaded:   {tex}", _GREEN))

    for tex, reason in result.textures_skipped:
        if verbose:
            print(_c(f"  Skipped:  {tex}", _YELLOW))
            print(_c(f"            {reason}", _DIM))
        else:
            print(_c(f"  Skipped:  {tex}", _YELLOW))

    for tex in result.textures_missing:
        print(_c(f"  Missing:  {tex}", _RED))

    for w in result.warnings:
        print(_c(f"  Warning:  {w}", _YELLOW))

    for e in result.errors:
        print(_c(f"  Error:    {e}", _RED))

    print(_c("  Done", _CYAN))


# ---------------------------------------------------------------------------
# Session summary
# ---------------------------------------------------------------------------
def log_summary(
    results: List[BuildResult],
    skipped_no_json: List[str],
) -> None:
    """Print an aggregated summary after all materials have been processed."""
    sep = _c("=" * 56, _BOLD)
    print()
    print(sep)
    print(_c("  KHAZAN MATERIAL IMPORTER – SUMMARY", _BOLD))
    print(sep)

    total = len(results) + len(skipped_no_json)
    processed = len(results)
    dry_runs = sum(1 for r in results if r.was_dry_run)
    total_loaded = sum(len(r.textures_loaded) for r in results)
    total_missing = sum(len(r.textures_missing) for r in results)
    total_skipped = sum(len(r.textures_skipped) for r in results)
    total_warnings = sum(len(r.warnings) for r in results)
    total_errors = sum(len(r.errors) for r in results)

    print(f"  Total materials seen   : {total}")
    print(f"  Processed              : {processed}"
          + (f"  ({dry_runs} dry-run)" if dry_runs else ""))
    print(f"  No JSON match (skipped): {len(skipped_no_json)}")
    print(f"  Textures loaded        : {_c(str(total_loaded), _GREEN)}")
    print(f"  Textures skipped       : {_c(str(total_skipped), _YELLOW)}")
    print(f"  Textures missing       : {_c(str(total_missing), _RED)}")
    print(f"  Warnings               : {_c(str(total_warnings), _YELLOW)}")
    print(f"  Errors                 : {_c(str(total_errors), _RED)}")

    if skipped_no_json:
        print()
        print(_c("  Materials with no JSON match:", _YELLOW))
        for name in skipped_no_json:
            print(f"    – {name}")

    if total_errors > 0:
        print()
        print(_c("  Materials with errors:", _RED))
        for r in results:
            if r.errors:
                print(f"    {r.material_name}")
                for e in r.errors:
                    print(f"      {e}")

    print(sep)
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Session header
# ---------------------------------------------------------------------------
def log_session_start(
    texture_folder: str,
    material_folder: str,
    n_jsons: int,
    n_textures: int,
    n_blender_mats: int,
) -> None:
    """Print a header block at the start of a processing run."""
    sep = _c("=" * 56, _BOLD)
    print()
    print(sep)
    print(_c("  KHAZAN MATERIAL IMPORTER", _BOLD + _CYAN))
    print(sep)
    print(f"  Texture folder  : {texture_folder}")
    print(f"  Material folder : {material_folder}")
    print(f"  JSONs found     : {n_jsons}")
    print(f"  PNGs indexed    : {n_textures}")
    print(f"  Blender mats    : {n_blender_mats}")
    print(sep)
    sys.stdout.flush()
