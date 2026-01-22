"""
Project-specific configurations for AIND Analysis Framework Visualization App.

This module defines configurations for each supported project/collection.
Add new projects here to make them available in the app.
"""

from data import DynamicForagingDataLoader, GenericDataLoader

from .models import AppConfig, AssetConfig, DataTableConfig, QueryConfig, ScatterPlotConfig


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
    query=QueryConfig(
        default_days_back=365,  # Load only recent year by default
    ),
    scatter_plot=ScatterPlotConfig(
        x_column="n_trials",
        y_column="LPT_AIC",
        color_column="agent_alias",
        tooltip_image_column="S3_location",
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
        info_columns=[
            "processing.data_processes.code.parameters.plot_types",
            "processing.data_processes.code.parameters.channels",
            "processing.data_processes.end_date_time",
        ],
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
    scatter_plot=ScatterPlotConfig(
        tooltip_image_column="location",
    ),
    id_column="_id",
    session_date_column="processing.data_processes.end_date_time",
    subject_id_column=None,  # Not available in this collection structure
)


# Configuration for Dynamic Foraging Lifetime project
# Uses GenericDataLoader since there's no specific utility function
# Note: This collection has a similar structure to NM - output_parameters is empty
# Data is in processing.data_processes.code.parameters (analysis_name, analysis_tag, etc.)
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
        info_columns=[
            "processing.data_processes.code.parameters.analysis_name",
            "processing.data_processes.code.parameters.analysis_tag",
            "processing.data_processes.end_date_time",
        ],
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
    scatter_plot=ScatterPlotConfig(
        tooltip_image_column="location",
    ),
    id_column="_id",
    session_date_column="processing.data_processes.end_date_time",
    subject_id_column=None,  # Not available in this collection structure
)


# =============================================================================
# Project Registry
# =============================================================================
# Maps display names to (collection_name, AppConfig) tuples
# Add new projects here to make them available in the app

PROJECT_REGISTRY: dict[str, tuple[str, AppConfig]] = {
    "Dynamic Foraging Model Fitting": (
        "dynamic-foraging-model-fitting",
        DYNAMIC_FORAGING_MODEL_FITTING_CONFIG,
    ),
    "Dynamic Foraging NM": (
        "dynamic-foraging-nm",
        DYNAMIC_FORAGING_NM_CONFIG,
    ),
    "Dynamic Foraging Lifetime": (
        "dynamic-foraging-lifetime",
        DYNAMIC_FORAGING_LIFETIME_CONFIG,
    ),
}
