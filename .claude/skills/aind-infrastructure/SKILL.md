---
name: aind-infrastructure
description: Knowledge about AIND data infrastructure including MongoDB/DocumentDB access patterns, S3 asset storage, collection schemas, and query patterns. Use when working with AIND analysis results, querying metadata, or accessing stored assets.
---

# AIND Analysis Infrastructure

This skill captures knowledge about the Allen Institute for Neural Dynamics (AIND) data infrastructure for analysis results.

## Overview

AIND stores analysis results in a two-tier system:
1. **MongoDB (DocumentDB)**: Metadata and pointers to S3 assets
2. **S3**: Actual result files (figures, tables, videos)

## MongoDB Access

### Connection Pattern

Use `aind-data-access-api` for DocumentDB access:

```python
from aind_data_access_api.document_db import MetadataDbClient

client = MetadataDbClient(
    host="api.allenneuraldynamics.org",  # Public API endpoint
    database="analysis",                  # Analysis database
    collection="<collection-name>",       # Project-specific collection
)
```

### One Collection Per Project

Each analysis project has its own collection. Examples:
- `dynamic-foraging-model-fitting` - Foraging behavior MLE fitting
- Other projects follow same pattern

### Query Patterns

```python
# Basic query with projection
records = client.retrieve_docdb_records(
    filter_query={"status": "success"},
    projection={"_id": 1, "subject_id": 1, "S3_location": 1},
    paginate=False,  # Set True for large queries
)
```

### Common Fields

Most collections include:
- `_id`: Unique record identifier
- `subject_id`: Animal/subject identifier
- `session_date`: Date of recording
- `status`: Processing status ("success", "failed")
- `S3_location` or `location`: Path to S3 assets
- `analysis_time` / `analysis_datetime`: When analysis ran

### Two Pipeline Formats

AIND has two analysis pipeline formats:

1. **AIND Analysis Framework** (new):
   - Nested structure: `processing.data_processes.output_parameters.*`
   - S3 location in `location` field

2. **Prototype pipeline** (older):
   - Flat structure: fields at root level
   - S3 location constructed from `_id`

When querying, use MongoDB projection aliasing to normalize:
```python
projection = {
    "_id": 1,
    "subject_id": "$processing.data_processes.output_parameters.subject_id",  # New
    # OR
    "subject_id": 1,  # Old (field at root)
}
```

## S3 Access

### Public Buckets

Analysis results are in public S3 buckets (no auth needed):

```python
import s3fs

# Anonymous access for public buckets
fs = s3fs.S3FileSystem(anon=True)
```

### Common Bucket Paths

```python
S3_PATH_ANALYSIS_ROOT = "s3://aind-dynamic-foraging-analysis-prod-o5171v"
S3_PATH_BONSAI_ROOT = "s3://aind-behavior-data/foraging_nwb_bonsai_processed"
```

### Asset URL Construction

Convert S3 path to HTTPS for web display:

```python
def s3_to_https(s3_path: str) -> str:
    """Convert s3://bucket/key to https://bucket.s3.amazonaws.com/key"""
    if s3_path.startswith("s3://"):
        s3_path = s3_path[5:]
    bucket = s3_path.split("/")[0]
    key = "/".join(s3_path.split("/")[1:])
    return f"https://{bucket}.s3.amazonaws.com/{key}"
```

### Reading Files from S3

```python
import json
import pickle

# JSON files
with fs.open("s3://bucket/path/file.json") as f:
    data = json.load(f)

# Pickle files
with fs.open("s3://bucket/path/file.pkl", "rb") as f:
    df = pickle.load(f)

# Check existence
if fs.exists("s3://bucket/path/file.png"):
    # File exists
```

### Batch Operations

For multiple S3 reads, use ThreadPoolExecutor:

```python
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def fetch_file(s3_path):
    # ... fetch logic
    pass

with ThreadPoolExecutor(max_workers=10) as executor:
    results = list(tqdm(
        executor.map(fetch_file, paths),
        total=len(paths),
        desc="Fetching files"
    ))
```

## Using aind-analysis-arch-result-access

This package provides ready-made functions for specific analyses:

```python
# Install: pip install aind-analysis-arch-result-access
from aind_analysis_arch_result_access import get_mle_model_fitting

# Get foraging model fitting results
# IMPORTANT: At least one filter parameter is required!
df = get_mle_model_fitting(
    subject_id="778869",           # Filter by subject (recommended for fast loading)
    # session_date="2024-10-24",   # Or filter by date
    # agent_alias="QLearning...",  # Or filter by model type
    # from_custom_query={...},     # Or use custom MongoDB query
    if_include_metrics=True,
    if_include_latent_variables=False,  # Set False for faster loading
    if_download_figures=False,
)

# For querying all records, use a broad custom query:
df_all = get_mle_model_fitting(
    from_custom_query={"status": {"$exists": True}},  # Match all with status field
    if_include_metrics=True,
    if_include_latent_variables=False,
)
```

**Important notes:**
- Function requires at least one of: `subject_id`, `session_date`, `agent_alias`, or `from_custom_query`
- Queries both pipeline formats automatically and merges results
- `only_recent_version=True` (default) deduplicates by keeping most recent analysis
- Loading all records can be slow; filter by subject_id for prototyping

Key columns in returned DataFrame:
- `_id`: Record identifier
- `subject_id`, `session_date`: Session info
- `agent_alias`: Model type used
- `n_trials`: Number of trials
- `S3_location`: Path to result files (use for constructing asset URLs)
- `status`: "success" or "failed"
- `pipeline_source`: "aind analysis framework" or "han's analysis pipeline"
- Metrics: `log_likelihood`, `AIC`, `BIC`, `prediction_accuracy`, etc.

## Common Asset Types

Assets stored in S3 per record:
- `fitted_session.png` - Main result figure
- `docDB_record.json` - Full analysis results
- `original_results_*.json` - Raw output files
- Latent variables (q-values, RPE, etc.)

## Best Practices

1. **Filter early**: Use MongoDB queries to reduce data before pandas operations
2. **Batch S3 operations**: Use threading for multiple file reads
3. **Cache results**: Consider caching DataFrames for repeated queries
4. **Handle both formats**: Account for old and new pipeline structures
5. **Check S3 existence**: Assets may not exist for all records
