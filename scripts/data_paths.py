"""Locations and prerequisite errors for optional benchmark data."""
from pathlib import Path
from site_env import ROOT, settings


def data_dir():
    path = Path(settings().get("DGPP_DATA_DIR") or ROOT / "data").expanduser()
    return path if path.is_absolute() else ROOT / path


def require_file(path):
    path = Path(path).expanduser()
    if not path.is_file():
        raise SystemExit(f"missing benchmark input: {path}\n"
                         "Run python3 scripts/prepare_data.py download for datasets, or "
                         "prepare_data.py tokens for token IDs; use --help for custom paths.")
    return path
