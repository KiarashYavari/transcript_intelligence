# Transcript Intelligence

A transcript analytics and retrieval platform designed for analyzing organizational conversations across customer support, external business calls, and internal engineering discussions.

This project processes raw transcript datasets into a structured analytics warehouse, generates semantic metadata, performs sentiment and topic analysis, and exposes a Retrieval-Augmented Generation (RAG) pipeline for intelligent querying.

---

# Table of Contents

1. [Project Overview](#1-project-overview)
2. [Assignment Goals](#2-assignment-goals)
3. [System Architecture](#3-system-architecture)
4. [Dataset Overview](#4-dataset-overview)
5. [ETL & Parquet Pipeline](#5-etl--parquet-pipeline)
6. [Analytics Layer](#6-analytics-layer)
7. [RAG Architecture](#7-rag-architecture)
8. [Project Structure](#8-project-structure)
9. [Data Warehouse Schema](#9-data-warehouse-schema)
10. [Installation](#10-installation)
11. [Running the Pipeline](#11-running-the-pipeline)
12. [Running Analytics](#12-running-analytics)
13. [Running Quality Checks](#13-running-quality-checks)
14. [Building the RAG System](#14-building-the-rag-system)
15. [Example Queries](#15-example-queries)
16. [Example Outputs](#16-example-outputs)
17. [Design Decisions](#17-design-decisions)
18. [Future Improvements](#18-future-improvements)

---

# 1. Project Overview

Organizations generate large volumes of conversational data across:

- Customer support calls
- Customer success / account management calls
- Internal engineering meetings
- Escalation and planning sessions

The goal of this project is to transform raw transcript JSON files into a structured intelligence platform capable of:

- Topic categorization
- Sentiment analysis
- Transcript analytics
- Action item extraction
- Key moment detection
- Semantic retrieval using RAG

The system is designed as a lightweight local-first architecture using:

- Python
- Pandas
- Parquet
- ChromaDB
- Sentence Transformers

---

# 2. Assignment Goals

This project addresses the following objectives:

## 1. Topic Categorization

Build a pipeline that processes transcripts and categorizes them by topic/theme.

Implemented through:

- Topic extraction tables
- Topic metadata enrichment
- Retrieval-aware topic indexing
- Searchable transcript chunks by topic

## 2. Sentiment Analysis

Analyze sentiment across call types and identify trends.

Implemented through:

- Sentiment-aware transcript chunks
- Sentiment aggregation
- Negative/positive call exploration
- Metadata-aware RAG querying by sentiment

## 3. Additional Insights

Additional insight systems implemented:

- Action item extraction
- Key moment detection
- Speaker participation analytics
- Semantic retrieval across meetings
- Topic-aware search
- Stakeholder-oriented query examples

---

# 3. System Architecture

```text
Raw JSON Dataset
        │
        ▼
Readers Layer
        │
        ▼
Transformers Layer
        │
        ▼
Parquet Warehouse
        │
        ├── Analytics Layer
        │
        ├── Quality Validation Layer
        │
        └── RAG Layer
                │
                ▼
        Semantic Retrieval
                │
                ▼
        LLM-Ready Prompt Generation
```

At a high level, the project has three major layers:

1. **Data Engineering Layer**  
   Converts raw JSON transcript files into normalized parquet tables.

2. **Analysis Layer**  
   Reads parquet tables and generates analytics such as topics, sentiment, speaker participation, action items, and key moments.

3. **RAG Layer**  
   Converts transcript data into enriched retrieval documents, embeds them, stores them in ChromaDB, and retrieves relevant context for question answering.

---

# 4. Dataset Overview

The dataset contains approximately 100 meeting folders.

Each meeting folder contains several JSON files, such as:

```text
events.json
meeting-info.json
speaker-meta.json
speakers.json
summary.json
transcript.json
```

Each JSON file captures a different part of a meeting:

| File | Purpose |
|---|---|
| `meeting-info.json` | Meeting-level metadata |
| `transcript.json` | Transcript sentences/chunks |
| `events.json` | Participant join/leave events |
| `speakers.json` | Speaker information |
| `speaker-meta.json` | Speaker metadata |
| `summary.json` | Meeting summaries, action items, key moments, and topics |

The pipeline normalizes these heterogeneous JSON files into analytics-ready parquet tables.

---

# 5. ETL & Parquet Pipeline

## Pipeline Goals

The ETL pipeline is responsible for:

- Reading raw transcript JSON files
- Standardizing schemas
- Extracting transcript chunks
- Structuring speaker information
- Generating normalized parquet datasets
- Preparing retrieval-ready metadata

## ETL Architecture

```text
Raw JSON Files
    ↓
Readers
    ↓
Transformers
    ↓
Schema Validation
    ↓
Parquet Writers
    ↓
Analytics Warehouse
```

## Main ETL Responsibilities

### Readers

Readers load raw JSON files from each meeting folder.

They are responsible for:

- Finding files
- Loading JSON safely
- Handling missing files
- Returning raw Python dictionaries/lists

### Transformers

Transformers convert raw JSON data into structured tabular records.

They are responsible for:

- Cleaning fields
- Normalizing nested data
- Creating consistent IDs
- Extracting transcript chunks
- Extracting topics, action items, and key moments

### Parquet Writer

The writer layer saves transformed records into parquet files.

It is responsible for:

- Creating output directories
- Writing dataframe outputs
- Keeping table names consistent
- Avoiding duplicated write logic

### Pipeline Runner

The pipeline runner orchestrates the full process:

```text
read raw data → transform records → build dataframes → write parquet tables
```

## Generated Parquet Tables

| Table | Description |
|---|---|
| `meetings` | Meeting-level metadata |
| `transcript_chunks` | Individual transcript chunks/sentences |
| `participant_events` | Participant join/leave events |
| `speaker_segments` | Speaker-level transcript segments |
| `meeting_summaries` | Structured meeting summaries |
| `action_items` | Extracted action items |
| `key_moments` | Important meeting moments |
| `topics` | Extracted meeting topics |
| `speaker_map` | Speaker identity mapping |

---

# 6. Analytics Layer

The analytics layer provides exploratory and operational insights over the parquet warehouse.

It is intentionally read-only. It does not modify parquet files.

## Analytics Architecture

```text
Parquet Warehouse
        │
        ▼
Analytics Engine
        │
        ├── Meeting Analytics
        ├── Speaker Analytics
        ├── Topic Analytics
        ├── Sentiment Analytics
        └── Action Item Analytics
```

## Analysis Modules

```text
src/analysis/
├── inspect_tables.py
├── analytics_examples.py
└── quality_checks.py
```

## `inspect_tables.py`

Used to inspect generated parquet tables.

Typical checks include:

- Row counts
- Column names
- Data types
- Null values
- Duplicate rows
- Sample records

Run:

```bash
python -m src.analysis.inspect_tables
```

## `analytics_examples.py`

Provides example analytics over the parquet warehouse.

Implemented analytics include:

### Meeting Analytics

- Longest meetings
- Transcript volume per meeting
- Summary coverage

### Speaker Analytics

- Top speakers by segment count
- Most talkative speakers by word count

### Topic Analytics

- Most discussed topics
- Topic frequency

### Sentiment Analytics

- Sentiment distribution
- Confidence score statistics

### Operational Analytics

- Meetings with most action items
- Key moment statistics
- Participant activity report

Run:

```bash
python -m src.analysis.analytics_examples
```

## `quality_checks.py`

Validates the generated parquet warehouse.

Checks include:

- Missing values
- Duplicate rows
- Empty transcript sentences
- Invalid transcript durations
- Referential integrity between meetings and related tables

Run:

```bash
python -m src.analysis.quality_checks
```

---

# 7. RAG Architecture

The RAG system converts transcript data into retrieval-ready semantic documents.

## RAG Goals

- Semantic transcript search
- Metadata-aware retrieval
- Topic-aware querying
- Sentiment-aware retrieval
- Speaker-aware retrieval
- LLM-ready context generation

## RAG Architecture

```text
Parquet Warehouse
        │
        ▼
Document Builder
        │
        ▼
RAG Documents
        │
        ▼
Embedding Pipeline
        │
        ▼
ChromaDB Vector Store
        │
        ▼
Retriever
        │
        ▼
Prompt Builder
        │
        ▼
LLM-Ready Prompt
```

## RAG Components

```text
src/rag/
├── __init__.py
├── config.py
├── document_builder.py
├── embedding_pipeline.py
├── vector_store.py
├── retriever.py
├── prompt_builder.py
├── query_engine.py
└── rag_runner.py
```

## RAG Document Types

| Document Type | Source Table | Purpose |
|---|---|---|
| `transcript_chunk` | `transcript_chunks` | Fine-grained transcript retrieval |
| `meeting_summary` | `meeting_summaries` | High-level meeting context |
| `action_item` | `action_items` | Follow-up and task retrieval |
| `key_moment` | `key_moments` | Important meeting highlight retrieval |

## Retrieval Document Example

Each transcript chunk becomes an enriched retrieval document:

```python
{
    "id": "transcript_chunk::meeting_001::45",
    "text": "Speaker: Sarah Chen\nSentiment: negative\nTranscript: The customer expressed frustration about onboarding delays.",
    "metadata": {
        "document_type": "transcript_chunk",
        "meeting_id": "meeting_001",
        "chunk_index": 45,
        "speaker_name": "Sarah Chen",
        "sentiment": "negative",
        "topics": "onboarding, support",
        "call_type": "customer_support",
        "confidence_score": 0.94
    }
}
```

## RAG Query Flow

```text
User Question
    ↓
Embed Query
    ↓
Search ChromaDB
    ↓
Retrieve Relevant Documents
    ↓
Build Prompt
    ↓
Return Retrieved Context + LLM-Ready Prompt
```

The current implementation retrieves relevant documents and generates an LLM-ready prompt. A final LLM generation layer can be added later.

---

# 8. Project Structure

A representative project layout:

```text
src/
├── analysis/
│   ├── analytics_examples.py
│   ├── inspect_tables.py
│   └── quality_checks.py
│
├── pipeline/
│   └── run_pipeline.py
│
├── rag/
│   ├── __init__.py
│   ├── config.py
│   ├── document_builder.py
│   ├── embedding_pipeline.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── prompt_builder.py
│   ├── query_engine.py
│   └── rag_runner.py
│
├── readers/
├── transformers/
├── writers/
└── schemas/
```

Depending on packaging/import choices, the project may also contain a `processed/` package or namespace used by the pipeline constants.

---

# 9. Data Warehouse Schema

## `meetings`

Meeting-level metadata.

| Column | Description |
|---|---|
| `meeting_id` | Unique meeting identifier |
| `title` | Meeting title/name |
| `call_type` | Type of call, such as support, external, or internal |
| `duration_minutes` | Meeting duration in minutes |
| `start_time` | Meeting start timestamp |

## `transcript_chunks`

Primary table for transcript analytics and RAG retrieval.

| Column | Description |
|---|---|
| `meeting_id` | Meeting identifier |
| `chunk_index` | Sentence/chunk order within the meeting |
| `speaker_id` | Speaker identifier if available |
| `speaker_name` | Speaker display name |
| `sentence` | Transcript sentence/chunk text |
| `sentiment` | Sentiment label for the chunk |
| `start_time_seconds` | Chunk start time |
| `end_time_seconds` | Chunk end time |
| `confidence_score` | Transcript confidence score |

## `participant_events`

Participant activity events.

| Column | Description |
|---|---|
| `meeting_id` | Meeting identifier |
| `participant_name` | Participant name |
| `event_type` | Join/leave or related event |
| `timestamp` | Event timestamp |
| `time` | Event time offset if available |

## `speaker_segments`

Speaker-level transcript segmentation.

| Column | Description |
|---|---|
| `meeting_id` | Meeting identifier |
| `speaker_name` | Speaker name |
| `segment_index` | Segment order if available |
| `text` | Segment text if available |

## `meeting_summaries`

Meeting summaries extracted from source summary files.

| Column | Description |
|---|---|
| `meeting_id` | Meeting identifier |
| `summary` | Summary text or structured summary field |

## `action_items`

Extracted action items.

| Column | Description |
|---|---|
| `meeting_id` | Meeting identifier |
| `action_item` | Action item text |
| `assignee` | Optional owner |
| `due_date` | Optional due date |

## `key_moments`

Important moments identified in meetings.

| Column | Description |
|---|---|
| `meeting_id` | Meeting identifier |
| `key_moment` | Key moment text |
| `moment_index` | Optional ordering field |

## `topics`

Extracted meeting topics.

| Column | Description |
|---|---|
| `meeting_id` | Meeting identifier |
| `topic` | Topic/theme text |
| `topic_index` | Topic order within the meeting |

## `speaker_map`

Speaker identity mapping.

| Column | Description |
|---|---|
| `meeting_id` | Meeting identifier |
| `speaker_id` | Speaker identifier |
| `speaker_name` | Speaker display name |

---

# 10. Installation

## Clone Repository

```bash
git clone <repository_url>
cd transcript-intelligence
```

## Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

For Windows:

```bash
venv\Scripts\activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

If requirements are not yet finalized, install the core packages:

```bash
pip install pandas pyarrow chromadb sentence-transformers
```

Optional packages depending on your implementation:

```bash
pip install pydantic python-dotenv
```

---

# 11. Running the Pipeline

Build the parquet warehouse from the raw JSON dataset:

```bash
python -m processed.src.pipeline.run_pipeline
```

Expected result:

```text
Processed parquet tables are written to the configured processed data directory.
```

After running the pipeline, verify the generated parquet files:

```bash
python -m processed.src.analysis.inspect_tables
```

---

# 12. Running Analytics

Run the analytics examples:

```bash
python -m processed.src.analysis.analytics_examples
```

Example report sections:

```text
TRANSCRIPT INTELLIGENCE ANALYTICS REPORT
------------------------------------------------------------------------------------------
LONGEST MEETINGS
------------------------------------------------------------------------------------------
...

------------------------------------------------------------------------------------------
MOST DISCUSSED TOPICS
------------------------------------------------------------------------------------------
...

------------------------------------------------------------------------------------------
SENTIMENT
------------------------------------------------------------------------------------------
...
```

This module is read-only and does not modify parquet files.

---

# 13. Running Quality Checks

Run data validation:

```bash
python -m processed.src.analysis.quality_checks
```

Example output:

```text
DATA QUALITY REPORT
================================================================================
[PASS] Missing Values - meetings
Details: Total missing: 0
--------------------------------------------------------------------------------
[PASS] Duplicate Rows - transcript_chunks
Details: Duplicate rows: 0
--------------------------------------------------------------------------------
[PASS] Empty Transcript Sentences
Details: Empty sentences: 0
--------------------------------------------------------------------------------
```

Quality checks help confirm that the data is ready for analytics and RAG retrieval.

---

# 14. Building the RAG System

## Build Vector Database

Before querying, build the local ChromaDB vector store:

```bash
python -m processed.src.rag.rag_runner --build
```

Expected output:

```text
Vector database built successfully.
Documents stored: <number_of_documents>
```

## Ask a Question

```bash
python -m processed.src.rag.rag_runner --query "What are customers unhappy about?"
```

The current implementation returns:

1. Retrieved documents
2. Metadata for each source
3. An LLM-ready prompt

## Query With Metadata Filters

Filter by sentiment:

```bash
python -m processed.src.rag.rag_runner \
  --query "What product issues are mentioned?" \
  --sentiment negative
```

Filter by document type:

```bash
python -m processed.src.rag.rag_runner \
  --query "What follow-up tasks were created?" \
  --document-type action_item
```

Filter by call type:

```bash
python -m processed.src.rag.rag_runner \
  --query "What are support customers complaining about?" \
  --call-type customer_support
```

---

# 15. Example Queries

## Customer Support

```text
What are customers unhappy about?
```

```text
What onboarding issues are most common?
```

```text
Which support calls contain negative sentiment?
```

## Product Management

```text
What product features are customers asking for?
```

```text
What recurring product complaints appear in support calls?
```

```text
Which topics are associated with negative sentiment?
```

## Sales / Customer Success

```text
What topics correlate with renewal risk?
```

```text
Which customers mentioned adoption challenges?
```

```text
What objections came up during external calls?
```

## Engineering

```text
Which meetings discuss infrastructure problems?
```

```text
What engineering escalations were mentioned?
```

```text
What internal calls mention blockers?
```

## Leadership

```text
Which issues appear most frequently across call types?
```

```text
What trends should leadership pay attention to?
```

```text
What are the most important risks mentioned in recent calls?
```

---

# 16. Example Outputs

## Analytics Output Example

```text
MOST DISCUSSED TOPICS
--------------------------------------------------------------------------------
topic                  frequency
onboarding             42
billing                31
support delays         27
product bugs           24
```

## Sentiment Output Example

```text
SENTIMENT
--------------------------------------------------------------------------------
sentiment              count
neutral                830
positive               410
negative               185
```

## RAG Retrieval Example

```text
==========================================================================================
RETRIEVED DOCUMENTS
==========================================================================================

SOURCE 1
------------------------------------------------------------------------------------------
Distance: 0.42
Metadata: {
    'document_type': 'transcript_chunk',
    'meeting_id': 'meeting_001',
    'call_type': 'customer_support',
    'speaker_name': 'Sarah Chen',
    'sentiment': 'negative',
    'topics': 'onboarding, support delays'
}

Speaker: Sarah Chen
Sentiment: negative
Transcript:
The customer said onboarding took too long and support response times were slow.
```

## RAG Prompt Example

```text
==========================================================================================
RAG PROMPT
==========================================================================================

You are a Transcript Intelligence assistant for a B2B SaaS company.

Use only the provided context to answer the question.
If the context is insufficient, say so clearly.

Your answer should include:
1. Direct answer
2. Supporting evidence from the retrieved context
3. Business implication for stakeholders

Context:
[Source 1]
Document Type: transcript_chunk
Meeting ID: meeting_001
Call Type: customer_support
Speaker: Sarah Chen
Sentiment: negative
Topics: onboarding, support delays
Text:
Speaker: Sarah Chen
Sentiment: negative
Transcript: The customer said onboarding took too long and support response times were slow.

Question:
What are customers unhappy about?

Answer:
```

## Example Final Answer After Adding an LLM

The current implementation prepares the prompt. If an LLM generation layer is added, the answer could look like:

```text
Customers appear most unhappy about onboarding delays, slow support response times,
missing reporting features, and unresolved product issues.

Supporting evidence:
- Several retrieved transcript chunks mention onboarding taking too long.
- Support-related conversations include frustration with slow response times.
- Product-related calls reference missing reporting capabilities.

Business implication:
Support and product teams should prioritize onboarding improvements, reporting features,
and response-time SLAs because these issues may affect adoption, satisfaction, and renewal risk.
```

---

# 17. Design Decisions

## Why Normalize JSON Into Parquet?

The raw dataset is stored as nested JSON files across many folders. This is flexible but difficult to query consistently.

Parquet was chosen because it provides:

- Columnar storage
- Fast analytical reads
- Compression
- Easy interoperability with pandas, DuckDB, Spark, and data warehouses
- A clean boundary between raw data and analysis-ready data

## Why Separate Readers, Transformers, and Writers?

The ETL pipeline is separated into layers to keep responsibilities clear:

| Layer | Responsibility |
|---|---|
| Readers | Load raw files safely |
| Transformers | Convert raw JSON into structured records |
| Writers | Save final dataframes to parquet |
| Pipeline Runner | Orchestrate the full workflow |

This makes the system easier to test, debug, and extend.

## Why Keep Analytics Read-Only?

Analysis modules should not mutate the source parquet files.

This keeps the architecture safe:

```text
Pipeline/Writers = write layer
Analysis/RAG = read layer
```

This separation prevents accidental overwrites and makes analytics reproducible.

## Why Use ChromaDB?

ChromaDB was selected for the first RAG version because it is:

- Local-first
- Easy to install
- Good for take-home projects
- Supports metadata filtering
- Simple to inspect and rebuild

For production, this could be replaced with Qdrant, Weaviate, Pinecone, or PostgreSQL with pgvector.

## Why Use Sentence Transformers?

Sentence Transformers provide:

- Local embedding generation
- No external API dependency
- Good semantic retrieval quality
- Simple batching support
- Easy integration with ChromaDB

The default model used in the design is:

```text
sentence-transformers/all-MiniLM-L6-v2
```

## Why Metadata-Aware Retrieval?

Transcript search is more useful when semantic similarity is combined with structured filters.

Examples:

```text
Find negative customer support calls about onboarding.
Find action items from internal engineering meetings.
Find key moments related to renewal risk.
```

Metadata fields such as `sentiment`, `call_type`, `speaker_name`, `meeting_id`, and `document_type` make retrieval more precise.

## Why Return an LLM-Ready Prompt Instead of Calling an LLM Immediately?

The current RAG version focuses on the retrieval system first.

This is useful because:

- Retrieval quality can be tested independently
- No API key is required
- The system is easier to debug
- Any LLM provider can be added later

The prompt can be passed to OpenAI, Anthropic, Ollama, or another local model.

---

# 18. Future Improvements

## Add LLM Generation

Current state:

```text
question → retrieval → prompt
```

Future state:

```text
question → retrieval → prompt → LLM answer
```

Possible integrations:

- OpenAI API
- Anthropic API
- Ollama local models
- Hugging Face local models


## Add Hybrid Retrieval

Current retrieval is vector-only.

Future retrieval could combine:

- Vector search
- Keyword search
- BM25
- Metadata filters
- Cross-encoder reranking

## Add Evaluation

RAG quality could be evaluated using:

- Manually written test questions
- Expected source document IDs
- Retrieval recall@k
- Answer faithfulness checks
- Human review

## Add Incremental Processing

Currently, the pipeline can be run as a full rebuild.

Future improvements:

- Detect new meetings only
- Process changed files only
- Avoid re-embedding unchanged documents
- Store pipeline run metadata

## Add Production Orchestration

Possible orchestration tools:

- Airflow
- Prefect
- Dagster
- cron-based local runner

## Add Tests

Recommended test areas:

- Reader tests
- Transformer tests
- Parquet writer tests
- Quality check tests
- RAG document builder tests
- Retriever tests

Example structure:

```text
tests/
├── test_readers.py
├── test_transformers.py
├── test_quality_checks.py
├── test_document_builder.py
└── test_retriever.py
```

---

# Conclusion

This project demonstrates a complete transcript intelligence workflow:

- Raw JSON ingestion
- ETL pipeline design
- Analytics warehouse modeling
- Data quality validation
- Topic and sentiment analysis
- Action item and key moment extraction
- Semantic retrieval architecture
- Retrieval-Augmented Generation foundation

The resulting system provides a strong foundation for building a scalable transcript analytics and intelligent search platform for B2B SaaS teams.

It can support multiple stakeholders:

| Stakeholder | Value |
|---|---|
| Support Leaders | Understand recurring customer issues |
| Product Managers | Identify product complaints and feature requests |
| Engineering Leads | Detect escalations and technical blockers |
| Sales / CS Teams | Track renewal risks and adoption barriers |
| Executives | Monitor high-level trends across the organization |
