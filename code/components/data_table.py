"""Data table component using Tabulator."""

import logging
from typing import TYPE_CHECKING, List, Optional

import pandas as pd
import panel as pn

from .base import BaseComponent

if TYPE_CHECKING:
    from config import AppConfig
    from core.base_app import DataHolder

logger = logging.getLogger(__name__)


class DataTable(BaseComponent):
    """
    Component for displaying data in a Tabulator table.

    Supports row selection and updates the DataHolder's selected_record_ids
    when rows are selected.
    """

    def create(self) -> pn.viewable.Viewable:
        """
        Create a reactive data table that updates when filtered_df changes.

        Returns:
            A Panel binding that renders the table reactively
        """
        return pn.bind(
            self._render_table,
            df=self.data_holder.param.filtered_df,
            additional_cols=self.data_holder.param.additional_columns,
        )

    def _get_display_columns(self) -> List[str]:
        """Get the default display columns from config."""
        return self.config.data_table.display_columns if self.config else []

    def _render_table(
        self,
        df: pd.DataFrame,
        additional_cols: Optional[List[str]] = None,
    ) -> pn.viewable.Viewable:
        """
        Render the data table.

        Args:
            df: DataFrame to display
            additional_cols: Additional columns beyond defaults

        Returns:
            Tabulator widget or placeholder message
        """
        if df is None or df.empty:
            return pn.pane.Markdown("No data available")

        # Get default display columns that exist in the dataframe
        default_cols = [c for c in self._get_display_columns() if c in df.columns]

        # Get additional columns that exist in the dataframe
        if additional_cols is None:
            additional_cols = []
        additional_cols = [c for c in additional_cols if c in df.columns]

        # Combine default and additional columns
        display_cols = default_cols + additional_cols

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
