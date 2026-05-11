# src/analysis/inspect_tables.py

"""
inspect_tables.py
-----------------

Utility module for inspecting generated parquet tables.

This module helps:
1. Load parquet tables safely.
2. Print schema information.
3. Display sample rows.
4. Show null statistics.
5. Show duplicate statistics.
6. Inspect table sizes.

Usage:
    python -m src.analysis.inspect_tables
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from processed.src.pipeline.constants import PROCESSED_DIR


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


TABLE_FILES: dict[str, str] = {
    "meetings": "meetings.parquet",
    "transcript_chunks": "transcript_chunks.parquet",
    "participant_events": "participant_events.parquet",
    "speaker_segments": "speaker_segments.parquet",
    "meeting_summaries": "meeting_summaries.parquet",
    "action_items": "action_items.parquet",
    "key_moments": "key_moments.parquet",
    "topics": "topics.parquet",
    "speaker_map": "speaker_map.parquet",
}


class TableInspector:
    """
    Helper class for inspecting parquet datasets.
    """

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir

    def load_table(self, table_name: str) -> pd.DataFrame:
        """
        Load parquet table safely.

        Args:
            table_name: Logical table name.

        Returns:
            Loaded dataframe.
        """

        file_name = TABLE_FILES.get(table_name)

        if not file_name:
            raise ValueError(f"Unknown table name: {table_name}")

        table_path = self.base_dir / file_name

        if not table_path.exists():
            raise FileNotFoundError(
                f"Parquet file does not exist: {table_path}"
            )

        try:
            dataframe = pd.read_parquet(table_path)
            logger.info("Loaded table: %s", table_name)
            return dataframe

        except Exception as error:
            logger.exception(
                "Failed to load parquet table: %s",
                table_name,
            )
            raise error

    @staticmethod
    def print_basic_info(
        dataframe: pd.DataFrame,
        table_name: str,
    ) -> None:
        """
        Print dataframe overview.
        """

        print("\n" + "=" * 80)
        print(f"TABLE: {table_name.upper()}")
        print("=" * 80)

        print(f"Rows: {len(dataframe):,}")
        print(f"Columns: {len(dataframe.columns)}")
        print(
            f"Memory Usage: "
            f"{dataframe.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        )

        print("\nCOLUMN TYPES")
        print("-" * 80)
        print(dataframe.dtypes)

    @staticmethod
    def print_null_statistics(dataframe: pd.DataFrame) -> None:
        """
        Print null value statistics.
        """

        print("\nNULL VALUE STATISTICS")
        print("-" * 80)

        null_counts = dataframe.isnull().sum()
        null_percentages = (
            dataframe.isnull().mean() * 100
        ).round(2)

        null_report = pd.DataFrame(
            {
                "null_count": null_counts,
                "null_percentage": null_percentages,
            }
        )

        null_report = null_report[
            null_report["null_count"] > 0
        ].sort_values(by="null_count", ascending=False)

        if null_report.empty:
            print("No null values found.")
        else:
            print(null_report)

    @staticmethod
    def print_duplicate_statistics(dataframe: pd.DataFrame) -> None:
        """
        Print duplicate row statistics.
        """

        duplicate_count = dataframe.duplicated().sum()

        print("\nDUPLICATE ROW STATISTICS")
        print("-" * 80)
        print(f"Duplicate Rows: {duplicate_count:,}")

    @staticmethod
    def print_sample_rows(
        dataframe: pd.DataFrame,
        sample_size: int = 5,
    ) -> None:
        """
        Print sample rows.
        """

        print("\nSAMPLE ROWS")
        print("-" * 80)
        print(dataframe.head(sample_size))

    def inspect_table(self, table_name: str) -> None:
        """
        Perform full inspection on a parquet table.
        """

        dataframe = self.load_table(table_name)

        self.print_basic_info(dataframe, table_name)
        self.print_null_statistics(dataframe)
        self.print_duplicate_statistics(dataframe)
        self.print_sample_rows(dataframe)


def main() -> None:
    """
    Main execution entry point.
    """

    inspector = TableInspector(PROCESSED_DIR)

    for table_name in TABLE_FILES:
        try:
            inspector.inspect_table(table_name)

        except Exception as error:
            logger.error(
                "Inspection failed for table '%s': %s",
                table_name,
                error,
            )


if __name__ == "__main__":
    main()