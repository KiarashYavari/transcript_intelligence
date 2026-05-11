"""
run_pipeline.py

Main ETL orchestration script.

Pipeline stages:
1. Discover meeting folders
2. Validate folder structure
3. Read JSON files
4. Transform into normalized records
5. Convert records into DataFrames
6. Persist parquet tables

This file should remain orchestration-focused only.
"""

from pathlib import Path
import logging

from constants import (
    DATASET_DIR,
    PROCESSED_DIR,
)

from readers import (
    get_meeting_directories,
    validate_meeting_directory,
    load_json_file,
)

from transformers import (
    transform_meeting_info,
    transform_transcript_chunks,
    transform_participant_events,
    transform_speaker_segments,
    transform_meeting_summary,
    transform_action_items,
    transform_key_moments,
    transform_topics,
    transform_speaker_map,
)

from parquet_writer import (
    records_to_dataframe,
    write_parquet,
    validate_dataframe_not_empty,
)


# ============================================================
# LOGGING CONFIGURATION
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger(__name__)


# ============================================================
# MAIN PIPELINE
# ============================================================

def main() -> None:
    """
    Execute full ETL pipeline.
    """

    logger.info("Starting transcript intelligence pipeline.")

    meeting_records = []
    transcript_records = []
    participant_event_records = []
    speaker_segment_records = []
    meeting_summary_records = []
    action_item_records = []
    key_moment_records = []
    topic_records = []
    speaker_map_records = []

    meeting_directories = get_meeting_directories(
        DATASET_DIR
    )

    logger.info(
        "Discovered %d meeting directories.",
        len(meeting_directories),
    )

    # ========================================================
    # PROCESS EACH MEETING
    # ========================================================

    for meeting_dir in meeting_directories:

        meeting_id = meeting_dir.name

        logger.info(
            "Processing meeting: %s",
            meeting_id,
        )

        try:

            # ------------------------------------------------
            # VALIDATE FOLDER
            # ------------------------------------------------

            if not validate_meeting_directory(
                meeting_dir
            ):
                continue

            # ------------------------------------------------
            # LOAD FILES
            # ------------------------------------------------

            meeting_info_data = load_json_file(
                meeting_dir / "meeting-info.json"
            )

            transcript_data = load_json_file(
                meeting_dir / "transcript.json"
            )

            events_data = load_json_file(
                meeting_dir / "events.json"
            )

            speakers_data = load_json_file(
                meeting_dir / "speakers.json"
            )

            summary_data = load_json_file(
                meeting_dir / "summary.json"
            )

            speaker_meta_data = load_json_file(
                meeting_dir / "speaker-meta.json"
            )

            # ------------------------------------------------
            # TRANSFORM DATA
            # ------------------------------------------------

            meeting_records.append(
                transform_meeting_info(
                    meeting_info_data
                )
            )

            transcript_records.extend(
                transform_transcript_chunks(
                    meeting_id,
                    transcript_data,
                )
            )

            participant_event_records.extend(
                transform_participant_events(
                    meeting_id,
                    events_data,
                )
            )

            speaker_segment_records.extend(
                transform_speaker_segments(
                    meeting_id,
                    speakers_data,
                )
            )

            meeting_summary_records.append(
                transform_meeting_summary(
                    summary_data
                )
            )

            action_item_records.extend(
                transform_action_items(
                    meeting_id,
                    summary_data,
                )
            )

            key_moment_records.extend(
                transform_key_moments(
                    meeting_id,
                    summary_data,
                )
            )

            topic_records.extend(
                transform_topics(
                    meeting_id,
                    summary_data,
                )
            )

            speaker_map_records.extend(
                transform_speaker_map(
                    meeting_id,
                    speaker_meta_data,
                )
            )

        except Exception as exc:

            logger.exception(
                "Failed processing meeting: %s",
                meeting_id,
            )

            continue

    # ========================================================
    # BUILD DATAFRAMES
    # ========================================================

    logger.info("Building DataFrames.")

    dataframe_mapping = {
        "meetings": records_to_dataframe(
            meeting_records
        ),

        "transcript_chunks": records_to_dataframe(
            transcript_records
        ),

        "participant_events": records_to_dataframe(
            participant_event_records
        ),

        "speaker_segments": records_to_dataframe(
            speaker_segment_records
        ),

        "meeting_summaries": records_to_dataframe(
            meeting_summary_records
        ),

        "action_items": records_to_dataframe(
            action_item_records
        ),

        "key_moments": records_to_dataframe(
            key_moment_records
        ),

        "topics": records_to_dataframe(
            topic_records
        ),

        "speaker_map": records_to_dataframe(
            speaker_map_records
        ),
    }

    # ========================================================
    # WRITE PARQUET TABLES
    # ========================================================

    logger.info("Writing parquet tables.")

    for table_name, dataframe in dataframe_mapping.items():

        validate_dataframe_not_empty(
            dataframe,
            table_name,
        )

        output_path = (
            PROCESSED_DIR /
            f"{table_name}.parquet"
        )

        write_parquet(
            dataframe,
            output_path,
        )

    logger.info(
        "Pipeline completed successfully."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()