"""Column selector component for choosing additional table columns."""

import logging
from typing import TYPE_CHECKING, List

import pandas as pd
import panel as pn

from utils import get_url_param_list, update_url_param

from .base import BaseComponent

if TYPE_CHECKING:
    from config import AppConfig
    from core.base_app import DataHolder

logger = logging.getLogger(__name__)


class ColumnSelector(BaseComponent):
    """
    Component for selecting additional columns to display in the data table.

    Shows a multi-select widget with all available columns (excluding defaults)
    and updates the DataHolder's additional_columns when selection changes.
    """

    def __init__(self, data_holder: "DataHolder", config: "AppConfig"):
        """Initialize the column selector."""
        super().__init__(data_holder, config)
        self.column_selector_widget = None
        self._initial_url_cols = get_url_param_list("cols")  # Read once on init

    def create(self) -> pn.viewable.Viewable:
        """
        Create a reactive column selector that updates when filtered_df changes.

        Returns:
            A Panel binding that renders the selector reactively
        """
        return pn.bind(  # type: ignore[return-value]
            self._render_selector,
            df=self.data_holder.param.filtered_df,
        )

    def _get_display_columns(self) -> List[str]:
        """Get the default display columns from config."""
        return self.config.data_table.display_columns if self.config else []

    def _render_selector(self, df: pd.DataFrame) -> pn.Column:
        """
        Render the column selector.

        Args:
            df: DataFrame to get columns from

        Returns:
            Column selector panel
        """
        # Get default columns
        default_cols = set(self._get_display_columns())

        # Get available columns (excluding defaults), sorted case-insensitively
        if df is not None and not df.empty:
            available_cols = [col for col in df.columns if col not in default_cols]
            available_cols = sorted(available_cols, key=self._column_sort_key)
        else:
            available_cols = []

        # Apply initial URL columns if they exist in available columns
        initial_value = [c for c in self._initial_url_cols if c in available_cols]

        # Create multi-select widget
        self.column_selector_widget = pn.widgets.MultiSelect(
            name="",
            options=available_cols,
            value=initial_value,
            size=min(15, max(5, len(available_cols))),  # Dynamic size based on options
            sizing_mode="stretch_width",
        )

        # Update data_holder and URL when selection changes
        def on_column_change(event):
            self.data_holder.additional_columns = list(event.new)
            update_url_param("cols", list(event.new) if event.new else None)

        self.column_selector_widget.param.watch(on_column_change, "value")

        # Apply initial columns to data_holder
        if initial_value:
            self.data_holder.additional_columns = initial_value

        # Status message
        if available_cols:
            status_msg = f"*{len(available_cols)} columns available*"
        else:
            status_msg = "*Load data to see available columns*"

        return pn.Card(
            pn.pane.Markdown(status_msg),
            self.column_selector_widget,
            title="Show additional Columns",
            collapsed=True,
            sizing_mode="stretch_width",
        )

    @staticmethod
    def _column_sort_key(column: str) -> tuple[int, str]:
        """Sort analysis framework columns before df_session columns."""
        is_session = column.startswith("df_session.")
        return (1 if is_session else 0, column.lower())
