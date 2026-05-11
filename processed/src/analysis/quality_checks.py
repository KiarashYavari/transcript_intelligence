"""
quality_checks.py
-----------------

Data quality validation module for parquet datasets.

Validates:
1. Missing values
2. Duplicate records
3. Empty transcript sentences
4. Referential integrity (new schema)
5. Invalid durations
6. Basic schema expectations

Usage:
    python -m src.analysis.quality_checks
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import pandas as pd

from processed.src.pipeline.constants import PROCESSED_DIR


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


@dataclass
class QualityCheckResult:
    check_name: str
    passed: bool
    details: str


class DataQualityChecker:
    """
    Data quality validation for transcript intelligence pipeline.
    """

    def __init__(self, processed_dir):
        self.processed_dir = processed_dir

        # Core tables (NEW SCHEMA)
        self.meetings_df = self._load_table("meetings.parquet")
        self.transcript_chunks_df = self._load_table("transcript_chunks.parquet")
        self.participant_events_df = self._load_table("participant_events.parquet")
        self.speaker_segments_df = self._load_table("speaker_segments.parquet")
        self.meeting_summaries_df = self._load_table("meeting_summaries.parquet")
        self.action_items_df = self._load_table("action_items.parquet")
        self.key_moments_df = self._load_table("key_moments.parquet")
        self.topics_df = self._load_table("topics.parquet")
        self.speaker_map_df = self._load_table("speaker_map.parquet")

    # -------------------------
    # Loader
    # -------------------------
    def _load_table(self, file_name: str) -> pd.DataFrame:
        file_path = self.processed_dir / file_name

        if not file_path.exists():
            raise FileNotFoundError(f"Missing file: {file_path}")

        try:
            df = pd.read_parquet(file_path)
            logger.info("Loaded: %s", file_name)
            return df
        except Exception as e:
            logger.exception("Failed loading: %s", file_name)
            raise e

    # -------------------------
    # Generic checks
    # -------------------------
    @staticmethod
    def check_missing_values(df: pd.DataFrame, name: str) -> QualityCheckResult:
        missing = df.isnull().sum().sum()

        return QualityCheckResult(
            check_name=f"Missing Values - {name}",
            passed=missing == 0,
            details=f"Total missing: {missing}",
        )

    @staticmethod
    def check_duplicates(df: pd.DataFrame, name: str) -> QualityCheckResult:
        dup = df.duplicated().sum()

        return QualityCheckResult(
            check_name=f"Duplicate Rows - {name}",
            passed=dup == 0,
            details=f"Duplicate rows: {dup}",
        )

    # -------------------------
    # Transcript checks
    # -------------------------
    def check_empty_transcripts(self) -> QualityCheckResult:
        if "sentence" not in self.transcript_chunks_df.columns:
            return QualityCheckResult(
                check_name="Empty Transcript Check",
                passed=False,
                details="Missing column: sentence",
            )

        empty = (
            self.transcript_chunks_df["sentence"]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
            .sum()
        )

        return QualityCheckResult(
            check_name="Empty Transcript Sentences",
            passed=empty == 0,
            details=f"Empty sentences: {empty}",
        )

    def check_invalid_durations(self) -> QualityCheckResult:
        if not {
            "start_time_seconds",
            "end_time_seconds",
        }.issubset(self.transcript_chunks_df.columns):

            return QualityCheckResult(
                check_name="Transcript Duration Check",
                passed=False,
                details="Missing time columns",
            )

        invalid = (
            self.transcript_chunks_df["end_time_seconds"]
            < self.transcript_chunks_df["start_time_seconds"]
        ).sum()

        return QualityCheckResult(
            check_name="Invalid Transcript Durations",
            passed=invalid == 0,
            details=f"Invalid durations: {invalid}",
        )

    # -------------------------
    # Referential integrity (NEW SCHEMA)
    # -------------------------
    def check_referential_integrity(self) -> List[QualityCheckResult]:
        results: List[QualityCheckResult] = []

        valid_meetings = set(self.meetings_df["meeting_id"])

        # transcript → meetings
        invalid_meetings = (
            ~self.transcript_chunks_df["meeting_id"].isin(valid_meetings)
        ).sum()

        results.append(
            QualityCheckResult(
                check_name="Transcript → Meetings FK",
                passed=invalid_meetings == 0,
                details=f"Invalid meeting refs: {invalid_meetings}",
            )
        )

        # speaker_segments consistency (if speaker_name exists)
        if "speaker_name" in self.speaker_segments_df.columns:
            invalid_speakers = (
                self.speaker_segments_df["speaker_name"].isna().sum()
            )

            results.append(
                QualityCheckResult(
                    check_name="Speaker Segment Validity",
                    passed=invalid_speakers == 0,
                    details=f"Missing speaker names: {invalid_speakers}",
                )
            )

        return results

    # -------------------------
    # Runner
    # -------------------------
    def run_all_checks(self) -> List[QualityCheckResult]:
        results: List[QualityCheckResult] = []

        tables = {
            "meetings": self.meetings_df,
            "transcript_chunks": self.transcript_chunks_df,
            "speaker_segments": self.speaker_segments_df,
            "action_items": self.action_items_df,
            "topics": self.topics_df,
        }

        for name, df in tables.items():
            results.append(self.check_missing_values(df, name))
            results.append(self.check_duplicates(df, name))

        results.extend(self.check_referential_integrity())
        results.append(self.check_empty_transcripts())
        results.append(self.check_invalid_durations())

        return results

    # -------------------------
    # Output
    # -------------------------
    @staticmethod
    def print_results(results: List[QualityCheckResult]) -> None:
        print("\nDATA QUALITY REPORT")
        print("=" * 80)

        for r in results:
            status = "PASS" if r.passed else "FAIL"
            print(f"[{status}] {r.check_name}")
            print(f"Details: {r.details}")
            print("-" * 80)


def main() -> None:
    checker = DataQualityChecker(PROCESSED_DIR)

    try:
        results = checker.run_all_checks()
        checker.print_results(results)

    except Exception as e:
        logger.exception("Quality check failed: %s", e)


if __name__ == "__main__":
    main()