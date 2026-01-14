"""
AIND Analysis Framework Explorer - Prototype

A Panel app for exploring analysis results from multiple AIND projects.

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

# Available project configurations with display names
PROJECT_OPTIONS = {
    "Dynamic Foraging Model Fitting": ("dynamic-foraging-model-fitting", DEFAULT_CONFIG),
    "Dynamic Foraging NM": ("dynamic-foraging-nm", DYNAMIC_FORAGING_NM_CONFIG),
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Panel extensions
pn.extension("tabulator")


class DataHolder(param.Parameterized):
    """Central state container for reactive updates."""

    selected_record_ids = param.List(default=[], doc="List of currently selected record IDs")
    filtered_df = param.DataFrame(default=pd.DataFrame(), doc="Filtered DataFrame")
    is_loaded = param.Boolean(default=False, doc="Whether data has been loaded")
    load_status = param.String(default="", doc="Status message from data loading")


class DynamicForagingApp(param.Parameterized):
    """
    Panel app for exploring AIND analysis results.

    This app demonstrates:
    - Project selection via dropdown
    - Deferred data loading (load on demand)
    - Data loading from MongoDB via aind-analysis-arch-result-access
    - Tabulator display with filtering
    - Asset viewing from S3
    - Reactive state management
    """

    # Current project configuration
    current_config = param.ClassSelector(class_=AppConfig, default=None, doc="Current project config")

    def __init__(self, **params):
        """
        Initialize the app.

        Args:
            **params: Optional parameters
        """
        super().__init__(**params)
        self.data_holder = DataHolder()
        self.df_full: pd.DataFrame = None
        self.asset_viewer: AssetViewer = None

        # Project selector widget
        self.project_selector = pn.widgets.Select(
            name="Select Project",
            options=list(PROJECT_OPTIONS.keys()),
            value="Dynamic Foraging Model Fitting",
            sizing_mode="stretch_width",
        )

        # Watch for project changes
        self.project_selector.param.watch(self._on_project_change, "value")

        # Initialize with default project config (but don't load data yet)
        self.current_config = PROJECT_OPTIONS[self.project_selector.value][1]
        self._init_asset_viewer()

    def _init_asset_viewer(self):
        """Initialize or reinitialize the asset viewer based on current config."""
        if self.current_config:
            self.asset_viewer = AssetViewer(
                s3_location_column=self.current_config.asset.s3_location_column,
                asset_filename=self.current_config.asset.asset_filename,
                width=self.current_config.asset.viewer_width,
            )

    def _on_project_change(self, event):
        """Handle project selection change."""
        project_name = event.new
        if project_name in PROJECT_OPTIONS:
            _, config = PROJECT_OPTIONS[project_name]
            self.current_config = config
            self._init_asset_viewer()

            # Reset data state when project changes
            self.data_holder.filtered_df = pd.DataFrame()
            self.data_holder.selected_record_ids = []
            self.data_holder.is_loaded = False
            self.data_holder.load_status = ""
            self.df_full = None

            logger.info(f"Project changed to: {project_name}")

    def _get_default_query(self) -> dict:
        """Get default DocDB query for recent data."""
        return self.current_config.query.get_default_query()

    def load_data(self, custom_query: dict = None) -> str:
        """Load data from the selected project collection."""
        if not self.current_config or not self.current_config.data_loader:
            return "Error: No data loader configured"

        query = custom_query if custom_query else self._get_default_query()

        try:
            logger.info(f"Loading data with query: {query}")
            self.data_holder.load_status = "Loading..."

            self.df_full = self.current_config.data_loader.load(query)

            if self.df_full is not None and not self.df_full.empty:
                logger.info(f"Loaded {len(self.df_full)} records")

                # Add asset URL column for hover tooltips
                s3_col = self.current_config.asset.s3_location_column
                asset_file = self.current_config.asset.asset_filename
                if s3_col in self.df_full.columns:
                    self.df_full["asset_url"] = self.df_full[s3_col].apply(
                        lambda s3: get_s3_image_url(f"{s3}/{asset_file}") if s3 else None
                    )

                self.data_holder.filtered_df = self.df_full.copy()
                self.data_holder.is_loaded = True
                self.data_holder.load_status = f"Loaded {len(self.df_full)} records"
                return f"Loaded {len(self.df_full)} records"
            else:
                logger.warning("No data returned from query")
                self.df_full = pd.DataFrame()
                self.data_holder.filtered_df = pd.DataFrame()
                self.data_holder.is_loaded = False
                self.data_holder.load_status = "No data returned from query"
                return "No data returned from query"

        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            self.df_full = pd.DataFrame()
            self.data_holder.filtered_df = pd.DataFrame()
            self.data_holder.is_loaded = False
            self.data_holder.load_status = f"Error: {e}"
            return f"Error: {e}"

    def _get_display_columns(self) -> list:
        """Columns to show in the main table."""
        return self.current_config.data_table.display_columns if self.current_config else []

    def apply_global_filter(self, query_string: str) -> str:
        """Apply pandas query filter to the data."""
        if not self.data_holder.is_loaded or self.df_full is None or self.df_full.empty:
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

    def create_project_selector(self) -> pn.Column:
        """Create the project selector panel."""
        load_button = pn.widgets.Button(
            name="Load Data",
            button_type="primary",
            sizing_mode="stretch_width",
        )
        status = pn.pane.Markdown("", css_classes=["alert", "alert-info", "p-2"])

        def load_callback(_event):
            result = self.load_data()
            status.object = result

        load_button.on_click(load_callback)

        return pn.Column(
            pn.pane.Markdown("### Project Selection"),
            self.project_selector,
            pn.pane.Markdown("*Select a project and click Load Data to begin*"),
            pn.layout.Spacer(height=10),
            load_button,
            status,
            sizing_mode="stretch_width",
        )

    def create_filter_panel(self) -> pn.Column:
        """Create the global filter panel."""
        filter_query = pn.widgets.TextAreaInput(
            name="Pandas Query",
            value="",
            placeholder=self.current_config.filter.default_placeholder if self.current_config else "",
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
        for ex in self.current_config.filter.example_queries:
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
            selectable=self.current_config.data_table.selectable,
            disabled=self.current_config.data_table.disabled,
            frozen_columns=self.current_config.data_table.frozen_columns,
            header_filters=self.current_config.data_table.header_filters,
            show_index=self.current_config.data_table.show_index,
            height=self.current_config.data_table.table_height,
            sizing_mode="stretch_width",
            stylesheets=[":host .tabulator {font-size: 11px;}"],
        )

        # Handle row selection
        def on_selection(event):
            if event.new:
                # Get IDs from all selected indices
                selected_ids = []
                for idx in event.new:
                    record_id = str(df.iloc[idx][self.current_config.id_column])
                    selected_ids.append(record_id)
                logger.info(f"Selected records: {selected_ids}")
                self.data_holder.selected_record_ids = selected_ids
            else:
                logger.info("No records selected")
                self.data_holder.selected_record_ids = []

        table.param.watch(on_selection, "selection")

        return table

    def create_welcome_content(self) -> pn.Column:
        """Create the welcome/placeholder content shown before data is loaded."""
        welcome_html = """
        <div style="text-align: center; padding: 50px 20px;">
            <h2>Welcome to AIND Analysis Framework Explorer</h2>
            <p style="font-size: 1.1em; color: #666;">
                Select a project from the sidebar and click <strong>Load Data</strong> to begin exploring.
            </p>
            <div style="margin-top: 30px; padding: 20px; background: #f8f9fa; border-radius: 8px;">
                <h3 style="margin-top: 0;">Available Projects:</h3>
                <ul style="text-align: left; max-width: 500px; margin: 0 auto;">
                    <li><strong>Dynamic Foraging Model Fitting</strong> - MLE model fitting results for behavioral data</li>
                    <li><strong>Dynamic Foraging NM</strong> - Neural modulation analysis results</li>
                    <li><em style="color: #888;">More projects coming soon...</em></li>
                </ul>
            </div>
            <p style="margin-top: 20px; color: #888; font-size: 0.9em;">
                This explorer supports multiple AIND analysis collections. Select a project to view its data.
            </p>
        </div>
        """

        return pn.Column(
            pn.pane.HTML(welcome_html, sizing_mode="stretch_width"),
            sizing_mode="stretch_width",
        )

    def create_main_content(self) -> pn.Column:
        """Create the main content area."""
        def render_content(is_loaded):
            """Render content based on whether data is loaded."""
            if not is_loaded:
                return self.create_welcome_content()

            # Data is loaded - show table and assets
            table = pn.bind(
                self.create_data_table,
                df=self.data_holder.param.filtered_df,
            )

            asset_display = self.asset_viewer.create_viewer(
                record_ids_param=self.data_holder.param.selected_record_ids,
                df_param=self.data_holder.param.filtered_df,
                id_column=self.current_config.id_column,
            )

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

        return pn.bind(render_content, is_loaded=self.data_holder.param.is_loaded)

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
                result = self.load_data(custom_query=query)
                status.object = result
            except json.JSONDecodeError as e:
                status.object = f"Invalid JSON: {e}"

        reload_button.on_click(reload_callback)

        # Build example queries from config
        examples = "\n**Example queries:**\n"
        for ex in self.current_config.query.get_example_queries():
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
            self.create_project_selector(),
            pn.layout.Divider(),
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
                    f"**Subjects:** {df[self.current_config.subject_id_column].nunique() if df is not None and self.current_config and self.current_config.subject_id_column in df.columns else 0}",
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

        template = pn.template.GoldenTemplate(title=self.current_config.app_title)

        # Add content to the template
        template.sidebar.append(sidebar_content)
        template.main.append(main_content)

        return template


# =============================================================================
# App Initialization
# =============================================================================

curdoc = curdoc()
curdoc.title = "AIND Analysis Framework Explorer"

# Create and serve the app
app = DynamicForagingApp()
layout = app.main_layout()
layout.servable()
