"""
AIND Analysis Framework Visualization App - Prototype

A Panel app for exploring analysis results from the dynamic-foraging-model-fitting collection.

To run:
    panel serve code/app.py --dev --show
"""

import logging

import pandas as pd
import panel as pn
import param
from bokeh.io import curdoc

# Import from aind-analysis-arch-result-access for data loading
from aind_analysis_arch_result_access import get_mle_model_fitting

from components.asset_viewer import AssetViewer, get_s3_image_url

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Panel extensions
pn.extension("tabulator")
curdoc().title = "Dynamic Foraging Model Fitting Explorer"


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

    def __init__(self):
        super().__init__()
        self.data_holder = DataHolder()
        self.df_full: pd.DataFrame = None

        # Asset viewer for displaying S3 figures
        self.asset_viewer = AssetViewer(
            s3_location_column="S3_location",
            asset_filename="fitted_session.png",
            width=900,
        )

        # Load data
        logger.info("Loading data from MongoDB...")
        self._load_data()

    def _load_data(self):
        """Load data from the dynamic-foraging-model-fitting collection."""
        try:
            # Use the existing result access package
            # For prototype: query single subject for faster loading
            self.df_full = get_mle_model_fitting(
                subject_id="778869",  # Single subject for fast prototype
                if_include_metrics=True,
                if_include_latent_variables=False,  # Skip for faster loading
                if_download_figures=False,
                paginate_settings={"paginate": False},
            )

            if self.df_full is not None:
                logger.info(f"Loaded {len(self.df_full)} records")

                # Add asset URL column for hover tooltips
                self.df_full["asset_url"] = self.df_full["S3_location"].apply(
                    lambda s3: get_s3_image_url(f"{s3}/fitted_session.png") if s3 else None
                )

                self.data_holder.filtered_df = self.df_full.copy()
            else:
                logger.warning("No data returned from query")
                self.df_full = pd.DataFrame()
                self.data_holder.filtered_df = pd.DataFrame()

        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            self.df_full = pd.DataFrame()
            self.data_holder.filtered_df = pd.DataFrame()

    def _get_display_columns(self) -> list:
        """Columns to show in the main table."""
        return [
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
            placeholder="e.g., subject_id == '730945' or n_trials > 100",
            height=60,
            sizing_mode="stretch_width",
        )

        filter_button = pn.widgets.Button(name="Apply Filter", button_type="primary", width=120)
        reset_button = pn.widgets.Button(name="Reset", button_type="light", width=80)
        status = pn.pane.Markdown("", css_classes=["alert", "alert-info", "p-2"])

        def apply_callback(event):
            result = self.apply_global_filter(filter_query.value)
            status.object = result

        def reset_callback(event):
            filter_query.value = ""
            result = self.apply_global_filter("")
            status.object = result

        filter_button.on_click(apply_callback)
        reset_button.on_click(reset_callback)

        examples = """
            **Example queries:**
            - `subject_id == '730945'`
            - `n_trials > 200`
            - `agent_alias.str.contains('QLearning')`
            - `prediction_accuracy > 0.6`
        """

        return pn.Column(
            pn.pane.Markdown("### Global Filter"),
            pn.pane.Markdown(examples),
            filter_query,
            pn.Row(filter_button, reset_button),
            status,
            width=350,
            css_classes=["card", "p-3", "m-2"],
        )

    def create_data_table(self, df: pd.DataFrame) -> pn.widgets.Tabulator:
        """Create the main data table."""
        if df is None or df.empty:
            return pn.pane.Markdown("No data available")

        # Get display columns that exist in the dataframe
        display_cols = [c for c in self._get_display_columns() if c in df.columns]

        table = pn.widgets.Tabulator(
            df[display_cols],
            selectable=True,  # Enable row selection (multi-select with Ctrl/Shift)
            disabled=True,  # Disable cell editing
            frozen_columns=["subject_id", "session_date", "agent_alias"],
            header_filters=True,
            show_index=False,
            height=400,
            sizing_mode="stretch_width",
            stylesheets=[":host .tabulator {font-size: 11px;}"],
        )

        # Handle row selection
        def on_selection(event):
            if event.new:
                # Get IDs from all selected indices
                selected_ids = []
                for idx in event.new:
                    record_id = str(df.iloc[idx]["_id"])
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
            id_column="_id",
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

    def create_sidebar(self) -> list:
        """Create sidebar content."""
        return [
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
                    f"**Subjects:** {df['subject_id'].nunique() if df is not None and 'subject_id' in df.columns else 0}",
                    css_classes=["alert", "alert-info", "p-2"],
                ),
                df=self.data_holder.param.filtered_df,
            ),
        ]

    def main_layout(self) -> pn.template.BootstrapTemplate:
        """Construct the full application layout."""
        filter_panel = self.create_filter_panel()
        main_content = self.create_main_content()

        template = pn.template.BootstrapTemplate(
            title="Dynamic Foraging Model Fitting Explorer",
            header_background="#0072B5",
            favicon="https://alleninstitute.org/wp-content/uploads/2021/10/cropped-favicon-32x32.png",
            main=[
                pn.Row(
                    filter_panel,
                    main_content,
                    sizing_mode="stretch_width",
                ),
            ],
            sidebar=self.create_sidebar(),
            theme="default",
        )
        template.sidebar_width = 200

        return template


# Create and serve the app
app = DynamicForagingApp()
layout = app.main_layout()
layout.servable()
