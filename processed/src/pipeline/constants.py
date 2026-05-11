"""
constants.py

Shared constants used throughout the pipeline.
"""

from pathlib import Path


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

DATASET_DIR = ROOT_DIR / "dataset"

PROCESSED_DIR = ROOT_DIR / "processed"


# ============================================================
# EXPECTED FILES
# ============================================================

EXPECTED_FILES = {
    "events.json",
    "meeting-info.json",
    "speaker-meta.json",
    "speakers.json",
    "summary.json",
    "transcript.json",
}