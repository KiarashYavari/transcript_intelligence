"""
parquet_writer.py

Handles:
- converting dataclass records into pandas DataFrames
- parquet persistence
- output directory creation
- parquet write validation
- logging

This module should NEVER:
- read raw JSON
- transform business logic
- manage dataset traversal
"""

from pathlib import Path
from dataclasses import asdict, is_dataclass
import logging
from typing import Any

import pandas as pd


logger = logging.getLogger(__name__)


# ============================================================
# DATAFRAME CONVERSION
# ============================================================

def records_to_dataframe(
    records: list[Any],
) -> pd.DataFrame:
    """
    Convert a list of dataclass records into a pandas DataFrame.

    Args:
        records:
            List of dataclass instances.

    Returns:
        pandas DataFrame

    Raises:
        ValueError:
            If records are not dataclass instances.
    """

    if not records:
        logger.warning(
            "Empty record list received. Returning empty DataFrame."
        )
        return pd.DataFrame()

    first_record = records[0]

    if not is_dataclass(first_record):
        raise ValueError(
            "records_to_dataframe expects dataclass instances."
        )

    return pd.DataFrame(
        [asdict(record) for record in records]
    )


# ============================================================
# PARQUET WRITER
# ============================================================

def write_parquet(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Persist DataFrame to parquet format.

    Args:
        dataframe:
            DataFrame to write.

        output_path:
            Destination parquet path.

    Raises:
        RuntimeError:
            If parquet writing fails.
    """

    try:

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        dataframe.to_parquet(
            output_path,
            index=False,
            engine="pyarrow",
        )

        logger.info(
            "Successfully wrote parquet file: %s",
            output_path,
        )

    except Exception as exc:

        logger.exception(
            "Failed writing parquet file."
        )

        raise RuntimeError(
            f"Failed to write parquet file: {output_path}"
        ) from exc


# ============================================================
# DATAFRAME VALIDATION
# ============================================================

def validate_dataframe_not_empty(
    dataframe: pd.DataFrame,
    table_name: str,
) -> None:
    """
    Log warning if dataframe is empty.

    Args:
        dataframe:
            DataFrame to validate.

        table_name:
            Human-readable table name.
    """

    if dataframe.empty:
        logger.warning(
            "DataFrame for table '%s' is empty.",
            table_name,
        )

    else:
        logger.info(
            "Table '%s' contains %d rows.",
            table_name,
            len(dataframe),
        )