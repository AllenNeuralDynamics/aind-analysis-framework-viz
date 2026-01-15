"""
Configuration dataclasses for AIND Analysis Framework Visualization App.

This module provides typed configuration classes for various app components.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data import DataLoader


@dataclass
class AssetConfig:
    """Configuration for S3 asset display."""

    s3_location_column: str = "S3_location"
    asset_filename: str = "fitted_session.png"
    viewer_width: int = 900
    info_columns: list[str] = field(
        default_factory=lambda: ["subject_id", "session_date", "agent_alias", "n_trials"]
    )

    def get_asset_url(self, s3_path: str | None, filename: str | None = None) -> str | None:
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

    display_columns: list[str] = field(
        default_factory=lambda: [
            "subject_id",
            "session_date",
            "agent_alias",
            "n_trials",
            "prediction_accuracy",
            "log_likelihood",
            "AIC",
            "BIC",
            "pipeline_source",
        ]
    )

    frozen_columns: list[str] = field(
        default_factory=lambda: ["subject_id", "session_date", "agent_alias"]
    )

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

        cutoff_date = (datetime.now() - timedelta(days=self.default_days_back)).strftime(
            "%Y-%m-%d"
        )

        # Support both pipeline formats:
        # - Old (prototype): session_date at root level
        # - New (AIND Analysis Framework): session_date nested in processing.data_processes
        return {
            "$or": [
                {"session_date": {"$gte": cutoff_date}},
                {
                    "processing.data_processes.output_parameters.session_date": {
                        "$gte": cutoff_date
                    }
                },
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
    example_queries: list[str] = field(
        default_factory=lambda: [
            "subject_id == '730945'",
            "n_trials > 200",
            "agent_alias.str.contains('QLearning')",
            "prediction_accuracy > 0.6",
        ]
    )


@dataclass
class ScatterPlotConfig:
    """Configuration for the interactive scatter plot component."""

    # Default axis columns (empty string = auto-select first numeric column)
    x_column: str = ""
    y_column: str = ""
    color_column: str = ""
    size_column: str = ""

    # Column for S3 image URLs in hover tooltips
    tooltip_image_column: str = ""

    # Visual defaults
    default_alpha: float = 0.7
    default_size: int = 10
    min_size: int = 5
    max_size: int = 40
    default_gamma: float = 1.0

    # Plot dimensions
    width: int = 600
    height: int = 500
    font_size: int = 12

    # Available color palettes
    color_palettes: list[str] = field(
        default_factory=lambda: [
            # Categorical
            "Category10",
            "Category20",
            "Category20b",
            "Category20c",
            "Set3",
            # Continuous (256)
            "Viridis256",
            "Plasma256",
            "Magma256",
            "Inferno256",
            "Cividis256",
            "Turbo256",
            "Greys256",
            "Blues256",
            "Greens256",
            "Oranges256",
            "Reds256",
            "Purples256",
        ]
    )

    # Maximum unique values for categorical color mapping
    max_categorical_values: int = 50


@dataclass
class AppConfig:
    """
    Main application configuration.

    Modify this class to adapt the app for different AIND projects.
    """

    # App metadata
    app_title: str = "AIND Analysis Framework Explorer"
    doc_title: str = "AIND Analysis Framework Explorer"

    # Data loader - implements the DataLoader interface
    # Each project should provide their own loader implementation
    data_loader: "DataLoader | None" = None

    # Sub-configurations
    asset: AssetConfig = field(default_factory=AssetConfig)
    data_table: DataTableConfig = field(default_factory=DataTableConfig)
    query: QueryConfig = field(default_factory=QueryConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    scatter_plot: ScatterPlotConfig = field(default_factory=ScatterPlotConfig)

    # ID column name (used for record selection)
    id_column: str = "_id"

    # Session date column (used for stats display)
    session_date_column: str = "session_date"

    # Subject ID column (used for stats display)
    subject_id_column: str | None = "subject_id"
