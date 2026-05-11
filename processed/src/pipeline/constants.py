"""
constants.py

Shared constants used throughout the pipeline.
"""

from pathlib import Path


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent

DATASET_DIR = ROOT_DIR.parents[2]  / "dataset"

PROCESSED_DIR = ROOT_DIR.parents[2] / "processed"


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