"""
transformers.py

Transforms raw JSON meeting data into normalized records.

Responsibilities:
- Convert raw JSON into strongly-typed dataclass records
- Normalize naming conventions
- Handle missing/invalid fields safely
- Keep transformation logic isolated from I/O

This module should NEVER:
- read files
- write parquet
- manage directories

Only transform data.
"""

from typing import Any

from schemas import (
    MeetingRecord,
    TranscriptChunkRecord,
    ParticipantEventRecord,
    SpeakerSegmentRecord,
    MeetingSummaryRecord,
    ActionItemRecord,
    KeyMomentRecord,
    TopicRecord,
    SpeakerMapRecord,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_str(value: Any, default: str = "") -> str:
    """
    Safely convert value to string.

    Args:
        value:
            Any incoming value.

        default:
            Default fallback value.

    Returns:
        String value.
    """

    if value is None:
        return default

    return str(value)


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert value to float.
    """

    try:
        return float(value)

    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """
    Safely convert value to integer.
    """

    try:
        return int(value)

    except (TypeError, ValueError):
        return default


# ============================================================
# MEETINGS TABLE
# ============================================================

def transform_meeting_info(
    raw_data: dict,
) -> MeetingRecord:
    """
    Transform meeting-info.json into MeetingRecord.

    Args:
        raw_data:
            Raw meeting-info.json data.

    Returns:
        MeetingRecord
    """

    return MeetingRecord(
        meeting_id=safe_str(raw_data.get("meetingId")),
        title=safe_str(raw_data.get("title")),
        organizer_email=safe_str(raw_data.get("organizerEmail")),
        host_email=safe_str(raw_data.get("host")),
        start_time=safe_str(raw_data.get("startTime")),
        end_time=safe_str(raw_data.get("endTime")),
        duration_minutes=safe_float(raw_data.get("duration")),
    )


# ============================================================
# TRANSCRIPT CHUNKS TABLE
# ============================================================

def transform_transcript_chunks(
    meeting_id: str,
    raw_data: dict,
) -> list[TranscriptChunkRecord]:
    """
    Transform transcript.json into transcript chunk records.

    Args:
        meeting_id:
            Meeting identifier.

        raw_data:
            Raw transcript.json data.

    Returns:
        List of TranscriptChunkRecord
    """

    records = []

    transcript_data = raw_data.get("data", [])

    for item in transcript_data:

        record = TranscriptChunkRecord(
            meeting_id=meeting_id,

            chunk_index=safe_int(item.get("index")),

            speaker_id=safe_int(item.get("speaker_id")),
            speaker_name=safe_str(item.get("speaker_name")),

            sentence=safe_str(item.get("sentence")),
            sentiment=safe_str(item.get("sentimentType")),

            start_time_seconds=safe_float(item.get("time")),
            end_time_seconds=safe_float(item.get("endTime")),

            confidence_score=safe_float(
                item.get("averageConfidence")
            ),
        )

        records.append(record)

    return records


# ============================================================
# PARTICIPANT EVENTS TABLE
# ============================================================

def transform_participant_events(
    meeting_id: str,
    raw_data: list[dict],
) -> list[ParticipantEventRecord]:
    """
    Transform events.json into participant event records.
    """

    records = []

    for item in raw_data:

        record = ParticipantEventRecord(
            meeting_id=meeting_id,

            participant_name=safe_str(
                item.get("participantName")
            ),

            event_type=safe_str(item.get("type")),

            event_timestamp_unix_ms=safe_int(
                item.get("timestamp")
            ),

            meeting_time_seconds=safe_float(
                item.get("time")
            ),
        )

        records.append(record)

    return records


# ============================================================
# SPEAKER SEGMENTS TABLE
# ============================================================

def transform_speaker_segments(
    meeting_id: str,
    raw_data: list[dict],
) -> list[SpeakerSegmentRecord]:
    """
    Transform speakers.json into speaker segment records.
    """

    records = []

    for item in raw_data:

        record = SpeakerSegmentRecord(
            meeting_id=meeting_id,

            speaker_name=safe_str(
                item.get("speakerName")
            ),

            start_time_seconds=safe_float(
                item.get("timestamp")
            ),

            end_time_seconds=safe_float(
                item.get("endTimeTs")
            ),
        )

        records.append(record)

    return records


# ============================================================
# MEETING SUMMARIES TABLE
# ============================================================

def transform_meeting_summary(
    raw_data: dict,
) -> MeetingSummaryRecord:
    """
    Transform summary.json into meeting summary record.
    """

    return MeetingSummaryRecord(
        meeting_id=safe_str(raw_data.get("meetingId")),

        summary_text=safe_str(raw_data.get("summary")),

        overall_sentiment=safe_str(
            raw_data.get("overallSentiment")
        ),

        sentiment_score=safe_float(
            raw_data.get("sentimentScore")
        ),
    )


# ============================================================
# ACTION ITEMS TABLE
# ============================================================

def transform_action_items(
    meeting_id: str,
    raw_data: dict,
) -> list[ActionItemRecord]:
    """
    Transform summary.json action items into records.
    """

    records = []

    action_items = raw_data.get("actionItems", [])

    for index, action_item in enumerate(action_items):

        record = ActionItemRecord(
            meeting_id=meeting_id,

            action_item_index=index,

            action_text=safe_str(action_item),
        )

        records.append(record)

    return records


# ============================================================
# KEY MOMENTS TABLE
# ============================================================

def transform_key_moments(
    meeting_id: str,
    raw_data: dict,
) -> list[KeyMomentRecord]:
    """
    Transform summary.json key moments into records.
    """

    records = []

    key_moments = raw_data.get("keyMoments", [])

    for index, item in enumerate(key_moments):

        record = KeyMomentRecord(
            meeting_id=meeting_id,

            key_moment_index=index,

            time_seconds=safe_float(item.get("time")),

            speaker_name=safe_str(
                item.get("speaker")
            ),

            moment_type=safe_str(item.get("type")),

            text=safe_str(item.get("text")),
        )

        records.append(record)

    return records


# ============================================================
# TOPICS TABLE
# ============================================================

def transform_topics(
    meeting_id: str,
    raw_data: dict,
) -> list[TopicRecord]:
    """
    Transform summary.json topics into records.
    """

    records = []

    topics = raw_data.get("topics", [])

    for index, topic in enumerate(topics):

        record = TopicRecord(
            meeting_id=meeting_id,

            topic_index=index,

            topic=safe_str(topic),
        )

        records.append(record)

    return records


# ============================================================
# SPEAKER MAP TABLE
# ============================================================

def transform_speaker_map(
    meeting_id: str,
    raw_data: dict,
) -> list[SpeakerMapRecord]:
    """
    Transform speaker-meta.json into speaker mapping records.

    Example:
        {
            "0": "Priya Patel",
            "1": "Yuki Tanaka"
        }
    """

    records = []

    for speaker_id, speaker_name in raw_data.items():

        record = SpeakerMapRecord(
            meeting_id=meeting_id,

            speaker_id=safe_int(speaker_id),

            speaker_name=safe_str(speaker_name),
        )

        records.append(record)

    return records