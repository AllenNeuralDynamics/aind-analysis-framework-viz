"""
AIND Analysis Framework Explorer - Prototype

A Panel app for exploring analysis results from multiple AIND projects.

To run:
    panel serve code/app.py --dev --show
"""

import logging

import pandas as pd
import panel as pn
import param
from bokeh.io import curdoc

from components import (
    AssetViewer,
    ColumnSelector,
    DataTable,
    DocDBQueryPanel,
    FilterPanel,
    ScatterPlot,
    StatsPanel,
    get_s3_image_url,
)
import __init__ as app_init
from config import (
    PROJECT_REGISTRY,
    AppConfig,
)
from core.base_app import BaseApp, DataHolder
from utils import get_url_param, get_url_param_list, update_url_param

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Panel extensions
pn.extension("tabulator")


class AINDAnalysisFrameworkApp(BaseApp):
    """
    Panel app for exploring AIND analysis results.

    Inherits from BaseApp to get:
    - DataHolder for reactive state management
    - Global filtering with pandas query
    - URL state synchronization helpers

    This app adds:
    - Project selection via dropdown (auto-loads data on selection)
    - Data loading from MongoDB via aind-analysis-arch-result-access
    - Tabulator display with filtering
    - Asset viewing from S3
    """

    # Placeholder value for project selector
    _PROJECT_PLACEHOLDER = "-- Select a project --"

    # Current project configuration
    current_config = param.ClassSelector(
        class_=AppConfig, default=None, doc="Current project config"
    )

    def __init__(self, **params):
        """
        Initialize the app.

        Args:
            **params: Optional parameters
        """
        super().__init__(**params)
        # BaseApp initializes: self.data_holder, self.df_full, self._components
        self.asset_viewer: AssetViewer = None

        # Project selector widget with placeholder
        self.project_selector = pn.widgets.Select(
            name="Select Project",
            options=[self._PROJECT_PLACEHOLDER] + list(PROJECT_REGISTRY.keys()),
            value=self._PROJECT_PLACEHOLDER,
            sizing_mode="stretch_width",
        )

        self.asset_columns_select = pn.widgets.IntSlider(
            name="Asset Columns",
            start=1,
            end=10,
            step=1,
            value=1,
            width=180,
        )
        self.asset_highlights_select = pn.widgets.MultiSelect(
            name="Highlights",
            options=[],
            value=[],
            size=6,
            width=220,
        )

        # Watch for project changes
        self.project_selector.param.watch(self._on_project_change, "value")
        self.data_holder.param.watch(self._update_highlight_options, "filtered_df")

        # Initialize with first project config (for components), but don't load data
        first_project = list(PROJECT_REGISTRY.keys())[0]
        self.current_config = PROJECT_REGISTRY[first_project][1]
        self.data_holder.table_height = self.current_config.data_table.table_height

        self.table_height_slider = pn.widgets.IntSlider(
            name="Table Height",
            start=200,
            end=1200,
            step=50,
            value=self.current_config.data_table.table_height,
            width=240,
        )
        self.table_height_slider.param.watch(self._on_table_height_change, "value")

        location = pn.state.location
        if location is not None:
            location.sync(self.asset_columns_select, {"value": "asset_cols"})
            location.sync(self.asset_highlights_select, {"value": "asset_hl"})
        self._init_components()

    def _init_components(self):
        """Initialize or reinitialize components based on current config."""
        if not self.current_config:
            return

        # Asset viewer
        self.asset_viewer = AssetViewer(
            s3_location_column=self.current_config.asset.s3_location_column,
            asset_filename=self.current_config.asset.asset_filename,
            width=self.current_config.asset.viewer_width,
            info_columns=self.current_config.asset.info_columns,
        )

        # Store component instances
        self._components["data_table"] = DataTable(self.data_holder, self.current_config)
        self._components["column_selector"] = ColumnSelector(self.data_holder, self.current_config)
        self._components["filter_panel"] = FilterPanel(
            self.data_holder,
            self.current_config,
            apply_filter_callback=self.apply_global_filter,
        )
        self._components["docdb_query"] = DocDBQueryPanel(
            self.data_holder,
            self.current_config,
            load_data_callback=self.load_data,
            get_default_query=self._get_default_query,
        )
        self._components["stats_panel"] = StatsPanel(self.data_holder, self.current_config)
        self._components["scatter_plot"] = ScatterPlot(self.data_holder, self.current_config)

    def _update_highlight_options(self, event) -> None:
        """Update highlight options when filtered_df changes.

        Handles the timing issue where:
        1. First call may have empty df (options=[]) but widget has URL values
        2. Second call has real data but widget was cleared by first call

        Solution: Don't clear widget value when options are empty.
        """
        df = event.new
        if df is None or df.empty:
            # Don't update options/value when df is empty - preserve URL-synced values
            return

        options = sorted(df.columns)

        # Get current widget value, but also check URL if widget is empty
        # (handles race condition where URL sync hasn't happened yet)
        current = list(self.asset_highlights_select.value)
        if not current:
            # URL sync may not have happened yet, read directly from URL
            current = get_url_param_list("asset_hl")

        # Filter to valid options
        current = [value for value in current if value in options]
        self.asset_highlights_select.options = options
        self.asset_highlights_select.value = current

    def _on_table_height_change(self, event) -> None:
        """Update table height when slider changes."""
        self.data_holder.table_height = int(event.new)

    def _on_project_change(self, event):
        """Handle project selection change - loads data immediately."""
        project_name = event.new

        # Prevent re-entry during project change (avoids URL sync loops)
        if getattr(self, "_changing_project", False):
            return
        self._changing_project = True

        try:
            self._handle_project_change(project_name)
        finally:
            self._changing_project = False

    def _handle_project_change(self, project_name: str):
        """Internal handler for project change."""
        # Handle placeholder selection
        if project_name == self._PROJECT_PLACEHOLDER:
            self.data_holder.filtered_df = pd.DataFrame()
            self.data_holder.selected_record_ids = []
            self.data_holder.additional_columns = []
            self.data_holder.is_loaded = False
            self.data_holder.load_status = ""
            self.df_full = None
            return

        if project_name in PROJECT_REGISTRY:
            # Check if this is initial load (from URL) or user switching projects
            is_initial_load = self.df_full is None

            _, config = PROJECT_REGISTRY[project_name]
            self.current_config = config
            self.data_holder.table_height = config.data_table.table_height
            self.table_height_slider.value = config.data_table.table_height

            # Update config in existing components (don't recreate to avoid URL sync issues)
            for component in self._components.values():
                component.config = config

            # Update asset viewer config
            if self.asset_viewer:
                self.asset_viewer.s3_location_column = config.asset.s3_location_column
                self.asset_viewer.asset_filename = config.asset.asset_filename
                self.asset_viewer.width = config.asset.viewer_width
                self.asset_viewer.info_columns = config.asset.info_columns

            # Reset data state before loading
            self.data_holder.filtered_df = pd.DataFrame()
            self.data_holder.selected_record_ids = []
            self.data_holder.additional_columns = []
            self.data_holder.is_loaded = False
            self.data_holder.load_status = ""
            self.df_full = None

            # On project switch (not initial load), clear filter and selection
            if not is_initial_load:
                filter_panel = self._components.get("filter_panel")
                if filter_panel and hasattr(filter_panel, "filter_query_widget"):
                    filter_panel.filter_query_widget.value = ""
                self._clear_table_selection()

                # Reset DocDB query to new project's default
                # With location.sync(), setting widget value will auto-update URL
                docdb_query_panel = self._components.get("docdb_query")
                if docdb_query_panel and hasattr(docdb_query_panel, "set_query"):
                    docdb_query_panel.set_query(self._get_default_query())

            logger.info(f"Project changed to: {project_name}")

            # Load data immediately
            # On initial load, check if docdb_query is in URL
            if is_initial_load:
                docdb_query_panel = self._components.get("docdb_query")
                url_query = None
                if docdb_query_panel and hasattr(docdb_query_panel, "get_current_query"):
                    url_query = docdb_query_panel.get_current_query()
                if docdb_query_panel and not get_url_param("docdb_query"):
                    docdb_query_panel.set_query(self._get_default_query())
                self.load_data(custom_query=url_query)
            else:
                self.load_data()

            # On initial load from URL, apply filter if present (preserve selection)
            if is_initial_load and self.data_holder.is_loaded:
                filter_panel = self._components.get("filter_panel")
                if filter_panel and hasattr(filter_panel, "filter_query_widget"):
                    filter_value = filter_panel.filter_query_widget.value
                    if filter_value:
                        self.apply_global_filter(filter_value, clear_selection=False)

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

    def apply_global_filter(self, query_string: str, clear_selection: bool = True) -> str:
        """Apply pandas query filter to the data.

        Args:
            query_string: Pandas query string to filter data
            clear_selection: If True, clears selected rows. Set to False when
                restoring state from URL to preserve selection.
        """
        if not self.data_holder.is_loaded or self.df_full is None or self.df_full.empty:
            return "No data loaded"

        if not query_string.strip():
            self.data_holder.filtered_df = self.df_full.copy()
            if clear_selection:
                self.data_holder.selected_record_ids = []
                self._clear_table_selection()
            return f"Reset to full dataset (N={len(self.df_full)})"

        try:
            logger.info(f"Applying filter query: '{query_string}'")
            filtered = self.df_full.query(query_string)
            if len(filtered) == 0:
                return "Query returned 0 results. Filter not applied."

            self.data_holder.filtered_df = filtered.copy()
            if clear_selection:
                self.data_holder.selected_record_ids = []
                self._clear_table_selection()
            return f"Showing {len(filtered)} of {len(self.df_full)} records"
        except Exception as e:
            logger.error(f"Query error: {e}")
            logger.error(f"Query string was: '{query_string}'")
            return f"Query error: {e}"

    def _clear_table_selection(self):
        """Clear the table widget's selection to update URL."""
        data_table = self._components.get("data_table")
        if data_table and data_table.table_widget is not None:
            data_table.table_widget.selection = []

    def create_project_selector(self) -> pn.Column:
        """Create the project selector panel."""
        return pn.Column(
            pn.pane.Markdown("### Project Selection"),
            self.project_selector,
            pn.pane.Markdown("*Select a project to load data automatically*"),
            sizing_mode="stretch_width",
        )

    def create_welcome_content(self) -> pn.Column:
        """Create the welcome/placeholder content shown before data is loaded."""
        welcome_html = """
        <div style="text-align: center; padding: 50px 20px;">
            <h2>Welcome to AIND Analysis Framework Explorer</h2>
            <p style="font-size: 1.1em; color: #666;">
                Select a project from the sidebar to begin exploring.
            </p>
            <div style="margin-top: 30px; padding: 20px; background: #f8f9fa; border-radius: 8px;">
                <h3 style="margin-top: 0;">Available Projects:</h3>
                <ul style="text-align: left; max-width: 500px; margin: 0 auto;">
                    <li><strong>Dynamic Foraging Model Fitting</strong> - MLE model fitting results for behavioral data</li>
                    <li><strong>Dynamic Foraging NM</strong> - Neural modulation analysis results</li>
                    <li><strong>Dynamic Foraging Lifetime</strong> - Lifetime analysis results</li>
                    <li><em style="color: #888;">More projects coming soon...</em></li>
                </ul>
            </div>
            <p style="margin-top: 30px; color: #aaa; font-size: 0.8em;">
                Built by Han Hou and Claude Code
            </p>
        </div>
        """

        return pn.Column(
            pn.pane.HTML(welcome_html, sizing_mode="stretch_width"),
            sizing_mode="stretch_width",
        )

    def create_main_content(self) -> pn.viewable.Viewable:
        """Create the main content area."""

        def render_content(is_loaded):
            """Render content based on whether data is loaded."""
            if not is_loaded:
                return self.create_welcome_content()

            # Data is loaded - show table, scatter plot, and assets
            table = self._components["data_table"].create()
            column_selector = self._components["column_selector"].create()
            scatter_plot = self._components["scatter_plot"].create()

            asset_display = self.asset_viewer.create_viewer(
                record_ids_param=self.data_holder.param.selected_record_ids,
                df_param=self.data_holder.param.filtered_df,
                id_column=self.current_config.id_column,
                columns_param=self.asset_columns_select,
                highlights_param=self.asset_highlights_select,
            )

            count_display = pn.bind(
                lambda df: pn.pane.Markdown(
                    f"**Showing {len(df) if df is not None else 0} records**",
                    css_classes=["alert", "alert-success", "p-2"],
                ),
                df=self.data_holder.param.filtered_df,
            )

            # Create tabs for different views
            tabs = pn.Tabs(
                (
                    "Data Table",
                    pn.Column(
                        pn.Row(
                            pn.pane.Markdown(
                                "*Click rows to select, or hold Ctrl/Cmd and click for multiple selections*"
                            ),
                            self.table_height_slider,
                            sizing_mode="stretch_width",
                        ),
                        table,
                        column_selector,
                        sizing_mode="stretch_width",
                    ),
                ),
                ("Scatter Plot", scatter_plot),
                sizing_mode="stretch_width",
            )
            pn.state.location.sync(tabs, {"active": "tab"})

            return pn.Column(
                count_display,
                tabs,
                pn.layout.Divider(),
                pn.pane.Markdown("### Selected Record Assets"),
                pn.Row(self.asset_columns_select, self.asset_highlights_select),
                asset_display,
                sizing_mode="stretch_width",
            )

        return pn.bind(render_content, is_loaded=self.data_holder.param.is_loaded)

    def create_sidebar(self) -> pn.Column:
        """Create sidebar content using extracted components."""
        credits = pn.pane.Markdown(
            (
                "---\n\n"
                "🔨 **Built by** Han Hou & Claude Code  \n"
                f"🏷️ **Version** {app_init.__version__}  \n"
                "🔗 **Repo** "
                "[aind-analysis-framework-viz](https://github.com/AllenNeuralDynamics/aind-analysis-framework-viz)"
            ),
            sizing_mode="stretch_width",
        )

        return pn.Column(
            self.create_project_selector(),
            self._components["docdb_query"].create(),  # Collapsed by default, for custom queries
            pn.layout.Divider(),
            self._components["filter_panel"].create(),
            pn.layout.Divider(),
            self._components["stats_panel"].create(),
            pn.Spacer(height=20),
            credits,
            sizing_mode="stretch_width",
        )

    def _sync_url_state(self):
        """Sync project selector to URL.

        Uses one-way sync to avoid bidirectional sync issues.
        Note: FilterPanel, ColumnSelector, and DataTable sync their own
        params (filter, cols, selected) in their create() methods.
        """
        # Read project from URL on initial load
        url_project = get_url_param("project")
        if url_project and url_project in PROJECT_REGISTRY:
            self.project_selector.value = url_project

        # One-way sync: widget → URL only
        def on_project_select(event):
            if event.new and event.new != self._PROJECT_PLACEHOLDER:
                update_url_param("project", event.new)

        self.project_selector.param.watch(on_project_select, "value")

    def main_layout(self) -> pn.template.GoldenTemplate:
        """Construct the full application layout."""
        main_content = self.create_main_content()
        sidebar_content = self.create_sidebar()

        template = pn.template.GoldenTemplate(
            title=self.current_config.app_title, favicon="static/favicon.svg"
        )

        # Add content to the template
        template.sidebar.append(sidebar_content)
        template.main.append(main_content)

        # Sync URL state after layout is created
        # Note: URL sync triggers _on_project_change which handles loading + filter
        self._sync_url_state()

        return template


# =============================================================================
# App Initialization
# =============================================================================

curdoc = curdoc()

# Create and serve the app
app = AINDAnalysisFrameworkApp()
curdoc.title = app.current_config.doc_title
layout = app.main_layout()
layout.servable()
