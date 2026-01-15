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
    LoadDataPanel,
    StatsPanel,
    get_s3_image_url,
)
from config import (
    PROJECT_REGISTRY,
    AppConfig,
)
from core.base_app import BaseApp, DataHolder

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
    - Project selection via dropdown
    - Deferred data loading (load on demand)
    - Data loading from MongoDB via aind-analysis-arch-result-access
    - Tabulator display with filtering
    - Asset viewing from S3
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
        # BaseApp initializes: self.data_holder, self.df_full, self._components
        self.asset_viewer: AssetViewer = None

        # Project selector widget
        self.project_selector = pn.widgets.Select(
            name="Select Project",
            options=list(PROJECT_REGISTRY.keys()),
            value="Dynamic Foraging Model Fitting",
            sizing_mode="stretch_width",
        )

        # Watch for project changes
        self.project_selector.param.watch(self._on_project_change, "value")

        # Initialize with default project config (but don't load data yet)
        self.current_config = PROJECT_REGISTRY[self.project_selector.value][1]
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
        self._components["load_panel"] = LoadDataPanel(
            self.data_holder,
            self.current_config,
            load_data_callback=self.load_data,
        )
        self._components["stats_panel"] = StatsPanel(self.data_holder, self.current_config)

    def _on_project_change(self, event):
        """Handle project selection change."""
        project_name = event.new
        if project_name in PROJECT_REGISTRY:
            _, config = PROJECT_REGISTRY[project_name]
            self.current_config = config
            self._init_components()

            # Reset data state when project changes
            self.data_holder.filtered_df = pd.DataFrame()
            self.data_holder.selected_record_ids = []
            self.data_holder.additional_columns = []
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
        return pn.Column(
            pn.pane.Markdown("### Project Selection"),
            self.project_selector,
            pn.pane.Markdown("*Select a project, optionally edit the DocDB query below, then click Load Data*"),
            sizing_mode="stretch_width",
        )

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

            # Data is loaded - show table and assets
            table = self._components["data_table"].create()

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

    def create_sidebar(self) -> pn.Column:
        """Create sidebar content using extracted components."""
        return pn.Column(
            self.create_project_selector(),
            self._components["docdb_query"].create(),
            self._components["load_panel"].create(),
            pn.layout.Divider(),
            self._components["column_selector"].create(),
            pn.layout.Divider(),
            self._components["filter_panel"].create(),
            pn.layout.Divider(),
            self._components["stats_panel"].create(),
            sizing_mode="stretch_width",
        )

    def _sync_url_state(self):
        """Centralize URL sync logic for the app.

        Syncs widgets to URL parameters for shareable links.
        Follows the two-way sync pattern from LCNE-patchseq-viz.

        Note: FilterPanel, ColumnSelector, and DataTable handle their own
        URL sync since their widgets are created during component initialization.
        """
        location = pn.state.location

        # Sync project selector to URL (only immediate widget in app)
        location.sync(self.project_selector, {'value': 'project'})

    def main_layout(self) -> pn.template.GoldenTemplate:
        """Construct the full application layout."""
        main_content = self.create_main_content()
        sidebar_content = self.create_sidebar()

        template = pn.template.GoldenTemplate(title=self.current_config.app_title)

        # Add content to the template
        template.sidebar.append(sidebar_content)
        template.main.append(main_content)

        # Sync URL state after layout is created
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
