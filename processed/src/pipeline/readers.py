"""
readers.py

Responsible for:
- reading JSON files
- validating meeting folders
- discovering dataset structure
"""

from pathlib import Path
import json
import logging
from typing import Any

from constants import EXPECTED_FILES


logger = logging.getLogger(__name__)


def load_json_file(file_path: Path) -> Any:
    """
    Safely load a JSON file.

    Args:
        file_path:
            Path to the JSON file.

    Returns:
        Parsed JSON content.

    Raises:
        FileNotFoundError:
            If file does not exist.

        ValueError:
            If JSON is invalid.
    """

    if not file_path.exists():
        raise FileNotFoundError(f"Missing file: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)

    except json.JSONDecodeError as exc:
        logger.exception("Invalid JSON detected.")
        raise ValueError(
            f"Invalid JSON in file: {file_path}"
        ) from exc


def get_meeting_directories(dataset_dir: Path) -> list[Path]:
    """
    Return all meeting directories in dataset folder.
    """

    return sorted(
        [
            path
            for path in dataset_dir.iterdir()
            if path.is_dir()
        ]
    )


def validate_meeting_directory(meeting_dir: Path) -> bool:
    """
    Validate required files exist in meeting folder.

    Args:
        meeting_dir:
            Path to meeting directory.

    Returns:
        True if valid, otherwise False.
    """

    existing_files = {
        file.name
        for file in meeting_dir.iterdir()
        if file.is_file()
    }

    missing_files = EXPECTED_FILES - existing_files

    if missing_files:
        logger.warning(
            "Skipping %s due to missing files: %s",
            meeting_dir.name,
            sorted(missing_files),
        )
        return False

    return True