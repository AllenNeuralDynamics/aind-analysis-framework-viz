"""
AIND Analysis Framework Visualization App - Prototype

A Panel app for exploring analysis results from the dynamic-foraging-model-fitting collection.

To run:
    panel serve code/app.py --dev --show
"""

import json
import logging
from datetime import datetime, timedelta

import pandas as pd
import panel as pn
import param
from bokeh.io import curdoc

from components.asset_viewer import AssetViewer, get_s3_image_url
from config import (
    DEFAULT_CONFIG,
    DYNAMIC_FORAGING_NM_CONFIG,
    AppConfig,
)

# Available project configurations
PROJECT_CONFIGS = {
    "default": DEFAULT_CONFIG,
    "dynamic-foraging-model-fitting": DEFAULT_CONFIG,
    "dynamic-foraging-nm": DYNAMIC_FORAGING_NM_CONFIG,
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Panel extensions
pn.extension("tabulator")


def get_config(project: str | None = None) -> AppConfig:
    """
    Get the configuration for the specified project.

    Args:
        project: Project name (key from PROJECT_CONFIGS).
                 If None, uses DEFAULT_CONFIG.

    Returns:
        AppConfig for the project
    """
    if project is None:
        return DEFAULT_CONFIG

    config = PROJECT_CONFIGS.get(project)
    if config is None:
        logger.warning(f"Unknown project '{project}', using DEFAULT_CONFIG")
        return DEFAULT_CONFIG

    return config


# Get project from URL parameter or use default
curdoc = curdoc()
curdoc.title = DEFAULT_CONFIG.doc_title


class DataHolder(param.Parameterized):
    """Central state container for reactive updates."""

    selected_record_ids = param.List(default=[], doc="List of currently selected record IDs")
    filtered_df = param.DataFrame(doc="Filtered DataFrame")


class DynamicForagingApp(param.Parameterized):
    """
    Panel app for exploring dynamic foraging model fitting results.

    This is a minimal prototype demonstrating:
    - Data loading from MongoDB via aind-analysis-arch-result-access
    - Tabulator display with filtering
    - Asset viewing from S3
    - Reactive state management
    """

    def __init__(self, config: AppConfig = DEFAULT_CONFIG):
        """
        Initialize the app with configuration.

        Args:
            config: Application configuration
        """
        super().__init__()
        self.config = config
        self.data_holder = DataHolder()
        self.df_full: pd.DataFrame = None

        # Asset viewer for displaying S3 figures
        self.asset_viewer = AssetViewer(
            s3_location_column=config.asset.s3_location_column,
            asset_filename=config.asset.asset_filename,
            width=config.asset.viewer_width,
        )

        # Load data
        logger.info("Loading data from MongoDB...")
        self._load_data()

    def _get_default_query(self) -> dict:
        """Get default DocDB query for recent 3 months of data.

        Queries both pipeline formats:
        - Old (prototype): session_date at root level
        - New (AIND Analysis Framework): session_date nested in processing.data_processes
        """
        return self.config.query.get_default_query()

    def _load_data(self, custom_query: dict = None) -> str:
        """Load data from the dynamic-foraging-model-fitting collection."""
        query = custom_query if custom_query else self._get_default_query()
        try:
            logger.info(f"Loading data with query: {query}")

            # Use configured data loader
            if self.config.data_loader is None:
                raise ValueError("No data loader configured in config.data_loader")

            self.df_full = self.config.data_loader.load(query)

            if self.df_full is not None and not self.df_full.empty:
                logger.info(f"Loaded {len(self.df_full)} records")

                # Add asset URL column for hover tooltips
                self.df_full["asset_url"] = self.df_full["S3_location"].apply(
                    lambda s3: get_s3_image_url(f"{s3}/fitted_session.png") if s3 else None
                )

                self.data_holder.filtered_df = self.df_full.copy()
                return f"Loaded {len(self.df_full)} records"
            else:
                logger.warning("No data returned from query")
                self.df_full = pd.DataFrame()
                self.data_holder.filtered_df = pd.DataFrame()
                return "No data returned from query"

        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            self.df_full = pd.DataFrame()
            self.data_holder.filtered_df = pd.DataFrame()
            return f"Error: {e}"

    def _get_display_columns(self) -> list:
        """Columns to show in the main table."""
        return self.config.data_table.display_columns

    def apply_global_filter(self, query_string: str) -> str:
        """Apply pandas query filter to the data."""
        if self.df_full is None or self.df_full.empty:
            return "No data loaded"

        if not query_string.strip():
            self.data_holder.filtered_df = self.df_full.copy()
            return f"Reset to full dataset (N={len(self.df_full)})"

        try:
            filtered = self.df_full.query(query_string)
            if len(filtered) == 0:
                return "Query returned 0 results. Filter not applied."

            self.data_holder.filtered_df = filtered.copy()
            return f"Showing {len(filtered)} of {len(self.df_full)} records"
        except Exception as e:
            return f"Query error: {e}"

    def create_filter_panel(self) -> pn.Column:
        """Create the global filter panel."""
        filter_query = pn.widgets.TextAreaInput(
            name="Pandas Query",
            value="",
            placeholder=self.config.filter.default_placeholder,
            height=60,
            sizing_mode="stretch_width",
        )

        filter_button = pn.widgets.Button(name="Apply Filter", button_type="primary", width=120)
        reset_button = pn.widgets.Button(name="Reset", button_type="light", width=80)
        status = pn.pane.Markdown("", css_classes=["alert", "alert-info", "p-2"])

        def apply_callback(_event):
            result = self.apply_global_filter(filter_query.value)
            status.object = result

        def reset_callback(_event):
            filter_query.value = ""
            result = self.apply_global_filter("")
            status.object = result

        filter_button.on_click(apply_callback)
        reset_button.on_click(reset_callback)

        # Build example queries from config
        examples = "\n**Example queries:**\n"
        for ex in self.config.filter.example_queries:
            examples += f"- `{ex}`\n"

        examples_card = pn.Card(
            pn.pane.Markdown(examples),
            title="Example Queries",
            collapsed=True,
        )

        return pn.Column(
            pn.pane.Markdown("### Global Filter"),
            filter_query,
            pn.Row(filter_button, reset_button),
            status,
            examples_card,
            sizing_mode="stretch_width",
        )

    def create_data_table(self, df: pd.DataFrame) -> pn.widgets.Tabulator:
        """Create the main data table."""
        if df is None or df.empty:
            return pn.pane.Markdown("No data available")

        # Get display columns that exist in the dataframe
        display_cols = [c for c in self._get_display_columns() if c in df.columns]

        table = pn.widgets.Tabulator(
            df[display_cols],
            selectable=self.config.data_table.selectable,
            disabled=self.config.data_table.disabled,
            frozen_columns=self.config.data_table.frozen_columns,
            header_filters=self.config.data_table.header_filters,
            show_index=self.config.data_table.show_index,
            height=self.config.data_table.table_height,
            sizing_mode="stretch_width",
            stylesheets=[":host .tabulator {font-size: 11px;}"],
        )

        # Handle row selection
        def on_selection(event):
            if event.new:
                # Get IDs from all selected indices
                selected_ids = []
                for idx in event.new:
                    record_id = str(df.iloc[idx][self.config.id_column])
                    selected_ids.append(record_id)
                logger.info(f"Selected records: {selected_ids}")
                self.data_holder.selected_record_ids = selected_ids
            else:
                logger.info("No records selected")
                self.data_holder.selected_record_ids = []

        table.param.watch(on_selection, "selection")

        return table

    def create_main_content(self) -> pn.Column:
        """Create the main content area."""
        # Reactive table that updates when filtered_df changes
        table = pn.bind(
            self.create_data_table,
            df=self.data_holder.param.filtered_df,
        )

        # Reactive asset viewer
        asset_display = self.asset_viewer.create_viewer(
            record_ids_param=self.data_holder.param.selected_record_ids,
            df_param=self.data_holder.param.filtered_df,
            id_column=self.config.id_column,
        )

        # Record count display
        count_display = pn.bind(
            lambda df: pn.pane.Markdown(
                f"**Showing {len(df) if df is not None else 0} records**",
                css_classes=["alert", "alert-success", "p-2"],
            ),
            df=self.data_holder.param.filtered_df,
        )

        return pn.Column(
            count_display,
            pn.pane.Markdown("### Records"),
            pn.pane.Markdown("*Click rows to select, or hold Ctrl/Cmd and click for multiple selections*"),
            table,
            pn.layout.Divider(),
            pn.pane.Markdown("### Selected Record Assets"),
            asset_display,
            sizing_mode="stretch_width",
        )

    def create_docdb_query_panel(self) -> pn.Column:
        """Create the DocDB query panel for data loading."""
        default_query = json.dumps(self._get_default_query(), indent=2)

        docdb_query = pn.widgets.TextAreaInput(
            name="DocDB Query (JSON)",
            value=default_query,
            placeholder='e.g., {"subject_id": "778869"}',
            height=100,
            sizing_mode="stretch_width",
        )

        reload_button = pn.widgets.Button(name="Reload Data", button_type="primary", width=120)
        status = pn.pane.Markdown("", css_classes=["alert", "alert-info", "p-2"])

        def reload_callback(_event):
            try:
                query = json.loads(docdb_query.value)
                result = self._load_data(custom_query=query)
                status.object = result
            except json.JSONDecodeError as e:
                status.object = f"Invalid JSON: {e}"

        reload_button.on_click(reload_callback)

        # Build example queries from config
        examples = "\n**Example queries:**\n"
        for ex in self.config.query.get_example_queries():
            examples += f"- `{ex}`\n"

        examples_card = pn.Card(
            pn.pane.Markdown(examples),
            title="Example Queries",
            collapsed=True,
        )

        return pn.Column(
            pn.pane.Markdown("### DocDB Query"),
            docdb_query,
            reload_button,
            status,
            examples_card,
            sizing_mode="stretch_width",
        )

    def create_sidebar(self) -> pn.Column:
        """Create sidebar content."""
        return pn.Column(
            self.create_docdb_query_panel(),
            pn.layout.Divider(),
            self.create_filter_panel(),
            pn.layout.Divider(),
            pn.pane.Markdown("### Selected"),
            pn.bind(
                lambda ids: pn.pane.Markdown(
                    f"**Count:** {len(ids)}" if ids else "**Count:** 0",
                    css_classes=["alert", "alert-secondary", "p-2"],
                ),
                ids=self.data_holder.param.selected_record_ids,
            ),
            pn.pane.Markdown("### Stats"),
            pn.bind(
                lambda df: pn.pane.Markdown(
                    f"**Records:** {len(df) if df is not None else 0}",
                    css_classes=["alert", "alert-info", "p-2"],
                ),
                df=self.data_holder.param.filtered_df,
            ),
            pn.bind(
                lambda df: pn.pane.Markdown(
                    f"**Subjects:** {df[self.config.subject_id_column].nunique() if df is not None and self.config.subject_id_column in df.columns else 0}",
                    css_classes=["alert", "alert-info", "p-2"],
                ),
                df=self.data_holder.param.filtered_df,
            ),
            sizing_mode="stretch_width",
        )

    def main_layout(self) -> pn.template.GoldenTemplate:
        """Construct the full application layout."""
        main_content = self.create_main_content()
        sidebar_content = self.create_sidebar()

        template = pn.template.GoldenTemplate(
            title=self.config.app_title,
        )
        
        # Add content to the template
        template.sidebar.append(sidebar_content)
        template.main.append(main_content)

        return template


# =============================================================================
# Project Selection via URL Parameter
# =============================================================================

# Get project from URL query parameter (e.g., ?project=dynamic-foraging-nm)
# If not specified, uses the default project
if pn.state.location is not None:
    _project_param = pn.state.location.query_params.get("project", "default")
else:
    # Fallback when not running in server context (e.g., testing)
    _project_param = "default"

_config = get_config(_project_param)

# Update doc title based on selected project
curdoc.title = _config.doc_title

# Create and serve the app with the selected config
app = DynamicForagingApp(config=_config)
layout = app.main_layout()
layout.servable()
