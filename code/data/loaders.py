"""
Data loader classes for AIND Analysis Framework Visualization App.

This module provides abstract and concrete data loader implementations
for fetching data from various sources (DocDB, custom APIs, etc.).
"""

import json
from abc import ABC, abstractmethod

import pandas as pd
import panel as pn

# Cache settings
CACHE_MAX_ITEMS = 10
CACHE_POLICY = "LRU"
CACHE_TTL = 3600  # 1 hour TTL to avoid stale data


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

    def clear_cache(self) -> None:
        """Clear the cache for this loader (if applicable)."""
        if hasattr(self, "_load_cached") and hasattr(self._load_cached, "clear"):
            self._load_cached.clear()


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
        """Load data using get_mle_model_fitting (cached)."""
        # Convert query dict to JSON string for hashable cache key
        query_key = json.dumps(query, sort_keys=True, default=str)
        return self._load_cached(
            query_key,
            self.include_metrics,
            self.include_latent_variables,
            self.download_figures,
            json.dumps(self.paginate_settings, sort_keys=True),
        )

    @staticmethod
    @pn.cache(max_items=CACHE_MAX_ITEMS, policy=CACHE_POLICY, ttl=CACHE_TTL)
    def _load_cached(
        query_key: str,
        include_metrics: bool,
        include_latent_variables: bool,
        download_figures: bool,
        paginate_settings_key: str,
    ) -> pd.DataFrame:
        """Cached data loading - memoized based on query and settings."""
        from aind_analysis_arch_result_access import get_mle_model_fitting

        query = json.loads(query_key)
        paginate_settings = json.loads(paginate_settings_key)

        df = get_mle_model_fitting(
            from_custom_query=query,
            if_include_metrics=include_metrics,
            if_include_latent_variables=include_latent_variables,
            if_download_figures=download_figures,
            paginate_settings=paginate_settings,
        )

        # Flatten params column if it exists and contains dicts
        if df is not None and "params" in df.columns:
            df = DynamicForagingDataLoader._flatten_params(df)

        return df

    @staticmethod
    def _flatten_params(df: pd.DataFrame) -> pd.DataFrame:
        """Flatten the params column into separate params.xxx columns."""
        if df["params"].apply(lambda x: isinstance(x, dict)).any():
            params_df = pd.json_normalize(df["params"].tolist(), sep=".")
            params_df.columns = [f"params.{col}" for col in params_df.columns]
            df = df.drop(columns=["params"]).join(params_df)
        return df


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

    def load(self, query: dict) -> pd.DataFrame:
        """Load data from DocDB and flatten using json_normalize (cached)."""
        # Convert query dict to JSON string for hashable cache key
        query_key = json.dumps(query, sort_keys=True, default=str)
        return self._load_cached(
            query_key,
            self.collection_name,
            self.host,
            self.database,
            self.flatten_separator,
            self.max_level,
            json.dumps(self.paginate_settings, sort_keys=True),
        )

    @staticmethod
    @pn.cache(max_items=CACHE_MAX_ITEMS, policy=CACHE_POLICY, ttl=CACHE_TTL)
    def _load_cached(
        query_key: str,
        collection_name: str,
        host: str,
        database: str,
        flatten_separator: str,
        max_level: int | None,
        paginate_settings_key: str,
    ) -> pd.DataFrame:
        """Cached data loading - memoized based on query and settings."""
        from aind_data_access_api.document_db import MetadataDbClient

        query = json.loads(query_key)
        paginate_settings = json.loads(paginate_settings_key)

        client = MetadataDbClient(
            host=host,
            database=database,
            collection=collection_name,
        )

        # Query DocDB - get all fields, no projection
        records = client.retrieve_docdb_records(
            filter_query=query,
            projection=None,
            **paginate_settings,
        )

        if not records:
            return pd.DataFrame()

        # Preprocess: extract single-element lists so json_normalize can flatten them
        for record in records:
            GenericDataLoader._extract_single_element_lists(record)

        # Use json_normalize for all flattening
        df = pd.json_normalize(
            records,
            sep=flatten_separator,
            max_level=max_level,
        )

        return df

    @staticmethod
    def _extract_single_element_lists(record: dict, max_depth: int = 3) -> None:
        """Recursively extract single-element lists from nested dict."""
        for key, value in list(record.items()):
            if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
                record[key] = value[0]
                GenericDataLoader._extract_single_element_lists(record[key], max_depth - 1)
            elif isinstance(value, dict) and max_depth > 0:
                GenericDataLoader._extract_single_element_lists(value, max_depth - 1)
