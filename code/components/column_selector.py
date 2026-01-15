"""Column selector component for choosing additional table columns."""

import logging
from typing import TYPE_CHECKING, List

import pandas as pd
import panel as pn

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

    def create(self) -> pn.viewable.Viewable:
        """
        Create a reactive column selector that updates when filtered_df changes.

        Returns:
            A Panel binding that renders the selector reactively
        """
        return pn.bind(
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
            available_cols = sorted(
                [col for col in df.columns if col not in default_cols],
                key=str.lower,
            )
        else:
            available_cols = []

        # Create multi-select widget
        column_selector = pn.widgets.MultiSelect(
            name="Additional Columns",
            options=available_cols,
            value=[],
            size=8,
            sizing_mode="stretch_width",
        )

        # Update data_holder when selection changes
        def on_column_change(event):
            self.data_holder.additional_columns = list(event.new)

        column_selector.param.watch(on_column_change, "value")

        # Status message
        if available_cols:
            status_msg = f"*{len(available_cols)} additional columns available*"
        else:
            status_msg = "*Load data to see available columns*"

        return pn.Column(
            pn.pane.Markdown("### Additional Columns"),
            pn.pane.Markdown(status_msg),
            column_selector,
            sizing_mode="stretch_width",
        )
