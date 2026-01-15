"""
Configuration module for AIND Analysis Framework Visualization App.

This module provides project-specific configuration settings, making the app
easily adaptable for different AIND projects by modifying only this file.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd


def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """
    Flatten a nested dictionary.

    Args:
        d: Dictionary to flatten
        parent_key: Parent key for nested recursion
        sep: Separator between keys

    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class DataLoader(ABC):
    """
    Abstract base class for data loaders.

    Each project should implement this interface to provide project-specific
    data loading logic while maintaining a consistent interface for the app.
    """

    @abstractmethod
    def load(self, query: dict) -> pd.DataFrame:
        """
        Load data from the data source.

        Args:
            query: MongoDB query as a dict

        Returns:
            DataFrame with the loaded data
        """
        pass


class DynamicForagingDataLoader(DataLoader):
    """
    Data loader for Dynamic Foraging Model Fitting project.

    Wraps the get_mle_model_fitting function from aind-analysis-arch-result-access.
    """

    def __init__(
        self,
        include_metrics: bool = True,
        include_latent_variables: bool = False,
        download_figures: bool = False,
        paginate_settings: dict | None = None,
    ):
        """
        Initialize the data loader.

        Args:
            include_metrics: Whether to include metrics in the returned data
            include_latent_variables: Whether to include latent variables
            download_figures: Whether to download figures
            paginate_settings: Pagination settings
        """
        self.include_metrics = include_metrics
        self.include_latent_variables = include_latent_variables
        self.download_figures = download_figures
        self.paginate_settings = paginate_settings or {"paginate": False}

    def load(self, query: dict) -> pd.DataFrame:
        """Load data using get_mle_model_fitting."""
        from aind_analysis_arch_result_access import get_mle_model_fitting

        return get_mle_model_fitting(
            from_custom_query=query,
            if_include_metrics=self.include_metrics,
            if_include_latent_variables=self.include_latent_variables,
            if_download_figures=self.download_figures,
            paginate_settings=self.paginate_settings,
        )


class GenericDataLoader(DataLoader):
    """
    Generic data loader for projects without existing utility functions.

    Queries DocDB directly and flattens nested fields from processing.data_processes.
    Uses pd.json_normalize for simple, efficient flattening.
    """

    def __init__(
        self,
        collection_name: str,
        host: str = "api.allenneuraldynamics.org",
        database: str = "analysis",
        flatten_separator: str = ".",
        max_level: int | None = None,
        paginate_settings: dict | None = None,
    ):
        """
        Initialize the generic data loader.

        Args:
            collection_name: Name of the DocDB collection
            host: DocDB host
            database: Database name
            flatten_separator: Separator for flattened keys (default: ".")
            max_level: Max depth to flatten (None = flatten all levels)
            paginate_settings: Pagination settings
        """
        self.collection_name = collection_name
        self.host = host
        self.database = database
        self.flatten_separator = flatten_separator
        self.max_level = max_level
        self.paginate_settings = paginate_settings or {"paginate": False}

        # Lazy initialization of client
        self._client = None

    def _get_client(self):
        """Get or create the DocDB client."""
        if self._client is None:
            from aind_data_access_api.document_db import MetadataDbClient

            self._client = MetadataDbClient(
                host=self.host,
                database=self.database,
                collection=self.collection_name,
            )
        return self._client

    def load(self, query: dict) -> pd.DataFrame:
        """Load data from DocDB and flatten using json_normalize."""
        client = self._get_client()

        # Query DocDB - get all fields, no projection
        records = client.retrieve_docdb_records(
            filter_query=query,
            projection=None,
            **self.paginate_settings,
        )

        if not records:
            return pd.DataFrame()

        # Preprocess: extract single-element lists so json_normalize can flatten them
        # This handles common DocDB pattern where processing.data_processes is [ {...} ]
        for record in records:
            self._extract_single_element_lists(record)

        # Use json_normalize for all flattening
        df = pd.json_normalize(
            records,
            sep=self.flatten_separator,
            max_level=self.max_level,
        )

        return df

    def _extract_single_element_lists(self, record: dict, max_depth: int = 3):
        """Recursively extract single-element lists from nested dict."""
        for key, value in list(record.items()):
            if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
                # Replace single-element list of dicts with the dict itself
                record[key] = value[0]
                # Recurse into the extracted dict
                self._extract_single_element_lists(record[key], max_depth - 1)
            elif isinstance(value, dict) and max_depth > 0:
                # Recurse into nested dicts
                self._extract_single_element_lists(value, max_depth - 1)


@dataclass
class AssetConfig:
    """Configuration for S3 asset display."""

    s3_location_column: str = "S3_location"
    asset_filename: str = "fitted_session.png"
    viewer_width: int = 900

    def get_asset_url(self, s3_path: Optional[str], filename: Optional[str] = None) -> Optional[str]:
        """
        Construct HTTPS URL for an S3 asset.

        Args:
            s3_path: S3 path from the record
            filename: Asset filename (uses default if not provided)

        Returns:
            HTTPS URL or None if s3_path is None
        """
        if not s3_path:
            return None

        filename = filename or self.asset_filename
        # Convert s3:// bucket/path to HTTPS URL
        # Format: s3://aind-data-bucket/path -> https://aind-data-bucket.s3.amazonaws.com/path
        if s3_path.startswith("s3://"):
            s3_path = s3_path[5:]  # Remove 's3://'
            bucket, *path_parts = s3_path.split("/", 1)
            path = path_parts[0] if path_parts else ""
            return f"https://{bucket}.s3.amazonaws.com/{path}/{filename}"

        return f"{s3_path}/{filename}"


@dataclass
class DataTableConfig:
    """Configuration for the main data table."""

    display_columns: list[str] = field(default_factory=lambda: [
        "subject_id",
        "session_date",
        "agent_alias",
        "n_trials",
        "prediction_accuracy",
        "log_likelihood",
        "AIC",
        "BIC",
        "pipeline_source",
    ])

    frozen_columns: list[str] = field(default_factory=lambda: [
        "subject_id",
        "session_date",
        "agent_alias",
    ])

    table_height: int = 400
    show_index: bool = False
    selectable: bool = True
    disabled: bool = True  # Disable cell editing
    header_filters: bool = True


@dataclass
class QueryConfig:
    """Configuration for data queries."""

    # Default query returns recent data
    default_days_back: int = 90
    # Optional override for default query (if None, uses get_default_query())
    default_query: dict | None = None

    def get_default_query(self) -> dict:
        """
        Get default DocDB query for recent data.

        Returns:
            MongoDB query as dict
        """
        # If default_query is explicitly set, use it
        if self.default_query is not None:
            return self.default_query

        cutoff_date = (datetime.now() - timedelta(days=self.default_days_back)).strftime("%Y-%m-%d")

        # Support both pipeline formats:
        # - Old (prototype): session_date at root level
        # - New (AIND Analysis Framework): session_date nested in processing.data_processes
        return {
            "$or": [
                {"session_date": {"$gte": cutoff_date}},
                {"processing.data_processes.output_parameters.session_date": {"$gte": cutoff_date}}
            ]
        }

    @staticmethod
    def get_example_queries() -> list[str]:
        """Example queries for the UI."""
        return [
            '{"subject_id": "778869"}',
            '{"session_date": {"$gte": "2024-10-01"}}',
            '{"n_trials": {"$gt": 200}}',
        ]


@dataclass
class FilterConfig:
    """Configuration for pandas query filtering."""

    default_placeholder: str = "e.g., subject_id == '730945' or n_trials > 100"
    example_queries: list[str] = field(default_factory=lambda: [
        "subject_id == '730945'",
        "n_trials > 200",
        "agent_alias.str.contains('QLearning')",
        "prediction_accuracy > 0.6",
    ])


@dataclass
class AppConfig:
    """Main application configuration.

    Modify this class to adapt the app for different AIND projects.
    """

    # App metadata
    app_title: str = "AIND Analysis Framework Explorer"
    doc_title: str = "AIND Analysis Framework Explorer"

    # Data loader - implements the DataLoader interface
    # Each project should provide their own loader implementation
    data_loader: DataLoader | None = None

    # Sub-configurations
    asset: AssetConfig = field(default_factory=AssetConfig)
    data_table: DataTableConfig = field(default_factory=DataTableConfig)
    query: QueryConfig = field(default_factory=QueryConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)

    # ID column name (used for record selection)
    id_column: str = "_id"

    # Session date column (used for stats display)
    session_date_column: str = "session_date"

    # Subject ID column (used for stats display)
    subject_id_column: str = "subject_id"


# =============================================================================
# Project Configurations
# =============================================================================

# Default configuration for Dynamic Foraging Model Fitting project
DYNAMIC_FORAGING_MODEL_FITTING_CONFIG = AppConfig(
    data_loader=DynamicForagingDataLoader(
        include_metrics=True,
        include_latent_variables=False,
        download_figures=False,
        paginate_settings={"paginate": False},
    ),
)


# Configuration for Dynamic Foraging NM (Neural Modulation) project
# Uses GenericDataLoader since there's no specific utility function
# Note: This collection has a different structure - output_parameters is empty
# Data is in processing.data_processes.code.parameters (plot_types, channels, fitted_model, etc.)
DYNAMIC_FORAGING_NM_CONFIG = AppConfig(
    app_title="AIND Analysis Framework Explorer - Dynamic Foraging NM",
    doc_title="AIND Analysis Framework Explorer - Dynamic Foraging NM",
    data_loader=GenericDataLoader(
        collection_name="dynamic-foraging-nm",
    ),
    asset=AssetConfig(
        s3_location_column="location",
        asset_filename="result.png",
        viewer_width=900,
    ),
    data_table=DataTableConfig(
        display_columns=[
            "_id",
            "processing.data_processes.code.parameters.plot_types",
            "processing.data_processes.code.parameters.channels",
            "processing.data_processes.code.parameters.fitted_model",
            "processing.data_processes.end_date_time",
        ],
        frozen_columns=["_id"],
    ),
    query=QueryConfig(
        default_query={},  # Empty query since this collection doesn't have session_date
        default_days_back=90,
    ),
    id_column="_id",
    # Note: session_date and subject_id are not available as separate fields in this collection
    # They would need to be parsed from file_location parameters if needed
    session_date_column="processing.data_processes.end_date_time",
    subject_id_column=None,  # Not available in this collection structure
)

# Configuration for Dynamic Foraging Lifetime project
# Uses GenericDataLoader since there's no specific utility function
# Note: This collection has a similar structure to NM - output_parameters is empty
# Data is in processing.data_processes.code.parameters (analysis_name, analysis_tag, last_n_days, etc.)
DYNAMIC_FORAGING_LIFETIME_CONFIG = AppConfig(
    app_title="AIND Analysis Framework Explorer - Dynamic Foraging Lifetime",
    doc_title="AIND Analysis Framework Explorer - Dynamic Foraging Lifetime",
    data_loader=GenericDataLoader(
        collection_name="dynamic-foraging-lifetime",
    ),
    asset=AssetConfig(
        s3_location_column="location",
        asset_filename="result.png",
        viewer_width=900,
    ),
    data_table=DataTableConfig(
        display_columns=[
            "_id",
            "processing.data_processes.code.parameters.analysis_name",
            "processing.data_processes.code.parameters.analysis_tag",
            "processing.data_processes.code.parameters.last_n_days",
            "processing.data_processes.end_date_time",
        ],
        frozen_columns=["_id"],
    ),
    query=QueryConfig(
        default_query={},  # Empty query since this collection doesn't have session_date
        default_days_back=90,
    ),
    id_column="_id",
    # Note: session_date and subject_id are not available as separate fields in this collection
    session_date_column="processing.data_processes.end_date_time",
    subject_id_column=None,  # Not available in this collection structure
)


# =============================================================================
# Usage Examples
# =============================================================================
#
# Example 1: Using a different collection with GenericDataLoader
#
# MY_COLLECTION_CONFIG = AppConfig(
#     app_title="My Collection Explorer",
#     doc_title="My Collection Explorer",
#     data_loader=GenericDataLoader(
#         collection_name="my-collection-name",
#         flatten_field="processing.data_processes",
#     ),
#     asset=AssetConfig(
#         s3_location_column="location",
#         asset_filename="my_figure.png",
#     ),
# )
#
# Example 2: Creating a custom data loader for a specific project
#
# class MyCustomDataLoader(DataLoader):
#     """Custom data loader with project-specific logic."""
#
#     def __init__(self, api_endpoint: str, timeout: int = 30):
#         self.api_endpoint = api_endpoint
#         self.timeout = timeout
#
#     def load(self, query: dict) -> pd.DataFrame:
#         # Custom implementation
#         from my_api import fetch_data
#         return fetch_data(
#             query=query,
#             endpoint=self.api_endpoint,
#             timeout=self.timeout,
#         )
#
# CUSTOM_LOADER_CONFIG = AppConfig(
#     app_title="AIND Analysis Framework Explorer - Custom API",
#     doc_title="AIND Analysis Framework Explorer - Custom API",
#     data_loader=MyCustomDataLoader(
#         api_endpoint="https://api.example.com/data",
#     ),
# )

