"""
analytics_examples.py
---------------------

Example analytical queries for the Transcript Intelligence platform.

This module demonstrates:
1. Meeting analytics
2. Speaker analytics
3. Transcript analytics
4. Topic analytics
5. Action item analytics
6. Key moment analytics

Usage:
    python -m src.analysis.analytics_examples
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


class AnalyticsEngine:
    """
    Analytics engine for transcript intelligence parquet datasets.
    """

    def __init__(self, processed_dir: Path):
        self.processed_dir = processed_dir

        # Core tables
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
            raise FileNotFoundError(f"Parquet file not found: {file_path}")

        try:
            df = pd.read_parquet(file_path)
            logger.info("Loaded: %s", file_name)
            return df
        except Exception as e:
            logger.exception("Failed loading: %s", file_name)
            raise e

    # -------------------------
    # Meeting Analytics
    # -------------------------
    def longest_meetings(self, top_n: int = 10) -> pd.DataFrame:
        if "duration_minutes" not in self.meetings_df.columns:
            raise ValueError("Missing duration_minutes column")

        return (
            self.meetings_df.sort_values(
                by="duration_minutes",
                ascending=False,
            )
            .head(top_n)
        )

    def summary_coverage_report(self) -> pd.DataFrame:
        total = self.meetings_df["meeting_id"].nunique()
        summarized = self.meeting_summaries_df["meeting_id"].nunique()

        return pd.DataFrame(
            {
                "metric": [
                    "total_meetings",
                    "meetings_with_summaries",
                    "coverage_percentage",
                ],
                "value": [
                    total,
                    summarized,
                    round((summarized / total * 100) if total else 0, 2),
                ],
            }
        )

    # -------------------------
    # Transcript Analytics
    # -------------------------
    def transcript_volume_per_meeting(self) -> pd.DataFrame:
        return (
            self.transcript_chunks_df.groupby("meeting_id")
            .size()
            .reset_index(name="chunk_count")
            .sort_values("chunk_count", ascending=False)
        )

    def transcript_word_statistics(self) -> pd.DataFrame:
        if "sentence" not in self.transcript_chunks_df.columns:
            raise ValueError("Missing sentence column")

        df = self.transcript_chunks_df.copy()

        df["word_count"] = df["sentence"].fillna("").astype(str).str.split().str.len()

        return pd.DataFrame(
            {
                "metric": [
                    "total_chunks",
                    "avg_words",
                    "max_words",
                    "min_words",
                    "total_words",
                ],
                "value": [
                    len(df),
                    round(df["word_count"].mean(), 2),
                    df["word_count"].max(),
                    df["word_count"].min(),
                    df["word_count"].sum(),
                ],
            }
        )

    def transcript_duration_statistics(self) -> pd.DataFrame:
        durations = (
            self.transcript_chunks_df["end_time_seconds"]
            - self.transcript_chunks_df["start_time_seconds"]
        )

        return pd.DataFrame(
            {
                "metric": [
                    "avg_chunk_duration",
                    "max_chunk_duration",
                    "min_chunk_duration",
                ],
                "value": [
                    round(durations.mean(), 2),
                    round(durations.max(), 2),
                    round(durations.min(), 2),
                ],
            }
        )

    # -------------------------
    # Speaker Analytics
    # -------------------------
    def top_speakers_by_segments(self, top_n: int = 10) -> pd.DataFrame:
        if "speaker_name" not in self.speaker_segments_df.columns:
            raise ValueError("Missing speaker_name column")

        return (
            self.speaker_segments_df.groupby("speaker_name")
            .size()
            .reset_index(name="segment_count")
            .sort_values("segment_count", ascending=False)
            .head(top_n)
        )

    def most_talkative_speakers(self, top_n: int = 10) -> pd.DataFrame:
        df = self.transcript_chunks_df.copy()

        df["word_count"] = df["sentence"].fillna("").astype(str).str.split().str.len()

        return (
            df.groupby("speaker_name")["word_count"]
            .sum()
            .reset_index()
            .sort_values("word_count", ascending=False)
            .head(top_n)
        )

    # -------------------------
    # Topic Analytics
    # -------------------------
    def most_discussed_topics(self, top_n: int = 10) -> pd.DataFrame:
        if "topic" not in self.topics_df.columns:
            raise ValueError("Missing topic column")

        return (
            self.topics_df.groupby("topic")
            .size()
            .reset_index(name="frequency")
            .sort_values("frequency", ascending=False)
            .head(top_n)
        )

    # -------------------------
    # Action Items
    # -------------------------
    def meetings_with_most_action_items(self, top_n: int = 10) -> pd.DataFrame:
        return (
            self.action_items_df.groupby("meeting_id")
            .size()
            .reset_index(name="action_item_count")
            .sort_values("action_item_count", ascending=False)
            .head(top_n)
        )

    # -------------------------
    # Participant Events
    # -------------------------
    def participant_activity_report(self) -> pd.DataFrame:
        if "event_type" not in self.participant_events_df.columns:
            raise ValueError("Missing event_type column")

        return (
            self.participant_events_df.groupby("event_type")
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )

    # -------------------------
    # Key Moments
    # -------------------------
    def key_moment_statistics(self) -> pd.DataFrame:
        return (
            self.key_moments_df.groupby("meeting_id")
            .size()
            .reset_index(name="key_moment_count")
            .sort_values("key_moment_count", ascending=False)
        )

    # -------------------------
    # Sentiment + Quality
    # -------------------------
    def sentiment_distribution(self) -> pd.DataFrame:
        if "sentiment" not in self.transcript_chunks_df.columns:
            raise ValueError("Missing sentiment column")

        return (
            self.transcript_chunks_df.groupby("sentiment")
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )

    def confidence_score_statistics(self) -> pd.DataFrame:
        if "confidence_score" not in self.transcript_chunks_df.columns:
            raise ValueError("Missing confidence_score column")

        s = self.transcript_chunks_df["confidence_score"]

        return pd.DataFrame(
            {
                "metric": [
                    "avg_confidence",
                    "min_confidence",
                    "max_confidence",
                ],
                "value": [
                    round(s.mean(), 4),
                    round(s.min(), 4),
                    round(s.max(), 4),
                ],
            }
        )


# -------------------------
# Runner
# -------------------------
def main() -> None:
    engine = AnalyticsEngine(PROCESSED_DIR)

    jobs = [
        ("LONGEST MEETINGS", engine.longest_meetings),
        ("TRANSCRIPT VOLUME", engine.transcript_volume_per_meeting),
        ("TOP SPEAKERS", engine.top_speakers_by_segments),
        ("ACTION ITEMS", engine.meetings_with_most_action_items),
        ("TOPICS", engine.most_discussed_topics),
        ("PARTICIPANT EVENTS", engine.participant_activity_report),
        ("KEY MOMENTS", engine.key_moment_statistics),
        ("SUMMARY COVERAGE", engine.summary_coverage_report),
        ("WORD STATS", engine.transcript_word_statistics),
        ("SENTIMENT", engine.sentiment_distribution),
        ("CONFIDENCE", engine.confidence_score_statistics),
        ("TALKATIVE SPEAKERS", engine.most_talkative_speakers),
        ("DURATION STATS", engine.transcript_duration_statistics),
    ]

    print("\n" + "=" * 90)
    print("TRANSCRIPT INTELLIGENCE ANALYTICS REPORT")
    print("=" * 90)

    for name, func in jobs:
        print("\n" + "-" * 90)
        print(name)
        print("-" * 90)

        try:
            print(func())
        except Exception as e:
            logger.exception("Failed: %s", name)
            print(f"ERROR: {e}")

    print("\nDONE")


if __name__ == "__main__":
    main()