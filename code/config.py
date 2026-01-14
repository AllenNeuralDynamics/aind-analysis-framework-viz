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

    def get_default_query(self) -> dict:
        """
        Get default DocDB query for recent data.

        Returns:
            MongoDB query as dict
        """
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
    app_name: str = "Dynamic Foraging Model Fitting Explorer"
    app_title: str = "Dynamic Foraging Model Fitting Explorer"
    doc_title: str = "Dynamic Foraging Model Fitting Explorer"

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


# Default configuration for Dynamic Foraging Model Fitting project
DEFAULT_CONFIG = AppConfig(
    data_loader=DynamicForagingDataLoader(
        include_metrics=True,
        include_latent_variables=False,
        download_figures=False,
        paginate_settings={"paginate": False},
    ),
)


# Example: Creating a config for a different project
#
# class MyProjectDataLoader(DataLoader):
#     """Custom data loader for my project."""
#
#     def __init__(self, custom_param: str, another_param: int):
#         self.custom_param = custom_param
#         self.another_param = another_param
#
#     def load(self, query: dict) -> pd.DataFrame:
#         # Custom loading logic with project-specific parameters
#         from my_package import get_my_data
#         return get_my_data(
#             query=query,
#             custom_option=self.custom_param,
#             numeric_option=self.another_param,
#         )
#
# MY_PROJECT_CONFIG = AppConfig(
#     app_name="My Project Explorer",
#     app_title="My Project Explorer",
#     doc_title="My Project Explorer",
#     data_loader=MyProjectDataLoader(
#         custom_param="value",
#         another_param=42,
#     ),
#     asset=AssetConfig(
#         s3_location_column="s3_path",
#         asset_filename="result.png",
#     ),
#     data_table=DataTableConfig(
#         display_columns=["col1", "col2", "col3"],
#     ),
# )

