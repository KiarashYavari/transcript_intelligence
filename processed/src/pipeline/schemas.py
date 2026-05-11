"""
schemas.py

Centralized schema definitions for the transcript intelligence pipeline.

These dataclasses represent normalized records that will later
be converted into DataFrames and stored as parquet tables.
"""

from dataclasses import dataclass
from typing import Optional


# ============================================================
# MEETING TABLE
# ============================================================

@dataclass(slots=True)
class MeetingRecord:
    """
    Represents high-level metadata about a meeting.
    """

    meeting_id: str
    title: str
    organizer_email: str
    host_email: str
    start_time: str
    end_time: str
    duration_minutes: float


# ============================================================
# TRANSCRIPT CHUNKS TABLE
# ============================================================

@dataclass(slots=True)
class TranscriptChunkRecord:
    """
    Represents a single utterance/chunk from the transcript.
    """

    meeting_id: str
    chunk_index: int

    speaker_id: int
    speaker_name: str

    sentence: str
    sentiment: str

    start_time_seconds: float
    end_time_seconds: float

    confidence_score: float


# ============================================================
# PARTICIPANT EVENTS TABLE
# ============================================================

@dataclass(slots=True)
class ParticipantEventRecord:
    """
    Represents participant join/leave activity.
    """

    meeting_id: str

    participant_name: str
    event_type: str

    event_timestamp_unix_ms: int
    meeting_time_seconds: float


# ============================================================
# SPEAKER SEGMENTS TABLE
# ============================================================

@dataclass(slots=True)
class SpeakerSegmentRecord:
    """
    Represents diarization/speaker timeline segments.
    """

    meeting_id: str

    speaker_name: str

    start_time_seconds: float
    end_time_seconds: float


# ============================================================
# MEETING SUMMARIES TABLE
# ============================================================

@dataclass(slots=True)
class MeetingSummaryRecord:
    """
    Represents overall meeting summary information.
    """

    meeting_id: str

    summary_text: str

    overall_sentiment: str
    sentiment_score: float


# ============================================================
# ACTION ITEMS TABLE
# ============================================================

@dataclass(slots=True)
class ActionItemRecord:
    """
    Represents extracted action items/tasks.
    """

    meeting_id: str

    action_item_index: int
    action_text: str


# ============================================================
# KEY MOMENTS TABLE
# ============================================================

@dataclass(slots=True)
class KeyMomentRecord:
    """
    Represents important timeline events extracted from the meeting.
    """

    meeting_id: str

    key_moment_index: int

    time_seconds: float

    speaker_name: Optional[str]

    moment_type: str
    text: str


# ============================================================
# TOPICS TABLE
# ============================================================

@dataclass(slots=True)
class TopicRecord:
    """
    Represents extracted meeting topics.
    """

    meeting_id: str

    topic_index: int
    topic: str


# ============================================================
# SPEAKER MAP TABLE
# ============================================================

@dataclass(slots=True)
class SpeakerMapRecord:
    """
    Maps speaker IDs to speaker names.
    """

    meeting_id: str

    speaker_id: int
    speaker_name: str