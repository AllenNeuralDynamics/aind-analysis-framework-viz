"""
Base application class with DataHolder pattern for reactive state management.

This module provides the foundational patterns for building Panel apps with:
- Centralized reactive state via DataHolder
- URL synchronization
- Global filtering
- Template layout helpers
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd
import panel as pn
import param

logger = logging.getLogger(__name__)


class DataHolder(param.Parameterized):
    """
    Centralized holder for reactive application state.

    Components can watch parameters on this object to react to changes.
    This pattern allows loose coupling between components while maintaining
    synchronized state.

    Attributes:
        selected_record_ids: List of currently selected record IDs
        filtered_df: The filtered DataFrame after applying global filters
        is_loaded: Whether data has been loaded
        load_status: Status message from data loading
        additional_columns: Additional columns to display beyond defaults
    """

    selected_record_ids = param.List(default=[], doc="List of currently selected record IDs")
    filtered_df = param.DataFrame(default=pd.DataFrame(), doc="Filtered DataFrame for display")
    is_loaded = param.Boolean(default=False, doc="Whether data has been loaded")
    load_status = param.String(default="", doc="Status message from data loading")
    additional_columns = param.List(default=[], doc="Additional columns to display beyond defaults")
    table_height = param.Integer(default=400, doc="Height of the data table")


class BaseApp(param.Parameterized):
    """
    Base class for Panel visualization apps.

    Provides common functionality:
    - DataHolder for reactive state
    - Global filtering with pandas query
    - URL state synchronization
    - Template layout helpers

    Subclasses should:
    1. Override `load_data()` to fetch project-specific data
    2. Override `create_main_content()` to build the UI
    3. Optionally override `get_display_columns()` for table display
    """

    def __init__(self, **params):
        super().__init__(**params)
        self.data_holder = DataHolder()
        self.df_full: Optional[pd.DataFrame] = None
        self._components: Dict[str, Any] = {}

    def load_data(self) -> pd.DataFrame:
        """
        Load the main DataFrame. Override in subclass.

        Returns:
            DataFrame with the project's data
        """
        raise NotImplementedError("Subclass must implement load_data()")

    def get_display_columns(self) -> list:
        """
        Return list of columns to display in the main table.
        Override to customize which columns appear in the table.

        Returns:
            List of column names
        """
        if self.df_full is not None:
            return list(self.df_full.columns)
        return []

    def get_id_column(self) -> str:
        """
        Return the name of the ID column for record selection.
        Override if your data uses a different ID column.

        Returns:
            Column name used as unique identifier
        """
        return "_id"

    def apply_global_filter(self, query_string: str) -> str:
        """
        Apply a pandas query filter to the data.

        Args:
            query_string: A pandas query string (e.g., "status == 'success'")

        Returns:
            Status message describing the result
        """
        if self.df_full is None:
            return "No data loaded"

        if not query_string.strip():
            self.data_holder.filtered_df = self.df_full.copy()
            return f"Reset to full dataset (N={len(self.data_holder.filtered_df)})"

        try:
            filtered = self.df_full.query(query_string)
            if len(filtered) == 0:
                return "Query returned 0 results. Filter not applied."

            self.data_holder.filtered_df = filtered
            return f"Query applied. {len(filtered)} records match (out of {len(self.df_full)})."
        except Exception as e:
            return f"Error in query: {str(e)}"

    def create_filter_panel(
        self,
        default_query: str = "",
        width: int = 400,
        examples: Optional[list] = None,
    ) -> pn.Column:
        """
        Create a global filter panel with query input.

        Args:
            default_query: Default query string
            width: Panel width
            examples: List of example queries to display

        Returns:
            Panel Column with filter controls
        """
        filter_query = pn.widgets.TextAreaInput(
            name="Query string",
            value=default_query,
            placeholder="Enter a pandas query string",
            sizing_mode="stretch_width",
            height=80,
        )

        filter_button = pn.widgets.Button(
            name="Apply filter",
            button_type="primary",
            width=120,
        )

        reset_button = pn.widgets.Button(
            name="Reset",
            button_type="light",
            width=80,
        )

        filter_status = pn.pane.Markdown("", css_classes=["alert", "p-2"])

        def apply_filter_callback(event):
            result = self.apply_global_filter(filter_query.value)
            if "error" in result.lower():
                filter_status.css_classes = ["alert", "alert-danger", "p-2"]
            else:
                filter_status.css_classes = ["alert", "alert-success", "p-2"]
            filter_status.object = result

        def reset_filter_callback(event):
            filter_query.value = ""
            result = self.apply_global_filter("")
            filter_status.css_classes = ["alert", "alert-success", "p-2"]
            filter_status.object = result

        filter_button.on_click(apply_filter_callback)
        reset_button.on_click(reset_filter_callback)

        # Build examples section if provided
        examples_md = ""
        if examples:
            examples_md = "**Examples:**\n" + "\n".join(f"- `{ex}`" for ex in examples)

        return pn.Column(
            pn.pane.Markdown("### Global Filter"),
            pn.pane.Markdown(examples_md) if examples_md else None,
            filter_query,
            pn.Row(filter_button, reset_button),
            filter_status,
            width=width,
            css_classes=["card", "p-3"],
        )

    def sync_url_state(self, widgets_mapping: Dict[str, tuple]) -> None:
        """
        Sync widget values to URL query parameters.

        Args:
            widgets_mapping: Dict mapping widget to (param_name, url_param_name)
                e.g., {widget: ("value", "my_param")}
        """
        location = pn.state.location
        for widget, (param_name, url_param) in widgets_mapping.items():
            location.sync(widget, {param_name: url_param})

    def create_main_content(self) -> pn.viewable.Viewable:
        """
        Create the main content layout. Override in subclass.

        Returns:
            Panel viewable object
        """
        raise NotImplementedError("Subclass must implement create_main_content()")

    def create_sidebar(self) -> list:
        """
        Create sidebar content. Override to customize.

        Returns:
            List of Panel objects for sidebar
        """
        return [
            pn.pane.Markdown("### Selected Records"),
            pn.bind(
                lambda ids: pn.pane.Markdown(
                    f"**Count:** {len(ids)}" if ids else "No records selected",
                    css_classes=["alert", "alert-secondary", "p-2"],
                ),
                ids=self.data_holder.param.selected_record_ids,
            ),
            pn.pane.Markdown("### Record Count"),
            pn.bind(
                lambda df: pn.pane.Markdown(
                    f"**{len(df) if df is not None else 0}** records",
                    css_classes=["alert", "alert-info", "p-2"],
                ),
                df=self.data_holder.param.filtered_df,
            ),
        ]

    def main_layout(
        self,
        title: str = "AIND Analysis Explorer",
        header_background: str = "#0072B5",
    ) -> pn.template.BootstrapTemplate:
        """
        Construct the full application layout.

        Args:
            title: Application title
            header_background: Header background color

        Returns:
            BootstrapTemplate ready to serve
        """
        # Load data
        self.df_full = self.load_data()
        self.data_holder.filtered_df = self.df_full.copy()

        # Create main content
        main_content = self.create_main_content()

        # Create template
        template = pn.template.BootstrapTemplate(
            title=title,
            header_background=header_background,
            favicon="https://alleninstitute.org/wp-content/uploads/2021/10/cropped-favicon-32x32.png",
            main=[main_content],
            sidebar=self.create_sidebar(),
            theme="default",
        )
        template.sidebar_width = 220

        return template
