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

    def __init__(self, data_holder: "DataHolder", config: "AppConfig"):
        """Initialize the data table."""
        super().__init__(data_holder, config)
        self.table_widget = None  # Exposed for URL sync
        self._current_df = None
        self._syncing_selection = False
        self.data_holder.param.watch(self._sync_selection_to_table, "selected_record_ids")

    def _apply_table_selection(self, selected_ids: List[str]) -> None:
        """Update table selection to match selected record IDs."""
        if self.table_widget is None or self._current_df is None:
            return
        id_values = self._current_df[self.config.id_column].astype(str).tolist()
        indices = [i for i, record_id in enumerate(id_values) if record_id in selected_ids]
        current = list(self.table_widget.selection or [])
        if set(current) == set(indices):
            return
        self._syncing_selection = True
        try:
            self.table_widget.selection = indices
        finally:
            self._syncing_selection = False

    def _sync_selection_to_table(self, event) -> None:
        """Sync DataHolder selection into the table widget."""
        selected_ids = [str(record_id) for record_id in (event.new or [])]
        self._apply_table_selection(selected_ids)

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

        self._current_df = df

        # Get default display columns that exist in the dataframe
        default_cols = [c for c in self._get_display_columns() if c in df.columns]

        # Get additional columns that exist in the dataframe
        if additional_cols is None:
            additional_cols = []
        additional_cols = [c for c in additional_cols if c in df.columns]

        # Combine default and additional columns
        display_cols = default_cols + additional_cols

        df_display = df[display_cols].copy()
        datetime_cols = df_display.select_dtypes(include=["datetime"]).columns
        for column in datetime_cols:
            df_display[column] = df_display[column].dt.strftime("%Y-%m-%d")

        self.table_widget = pn.widgets.Tabulator(
            df_display,
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
            if self._syncing_selection:
                return
            if event.new:
                # Get IDs from all selected indices
                selected_ids = []
                for idx in event.new:
                    # Bounds check to prevent index errors when filter changes
                    if idx < len(df):
                        record_id = str(df.iloc[idx][self.config.id_column])
                        selected_ids.append(record_id)
                logger.info(f"Selected records: {selected_ids}")
                if selected_ids != self.data_holder.selected_record_ids:
                    self.data_holder.selected_record_ids = selected_ids
            else:
                logger.info("No records selected")
                if self.data_holder.selected_record_ids:
                    self.data_holder.selected_record_ids = []

        self.table_widget.param.watch(on_selection, "selection")

        # Sync to URL for this table widget instance
        pn.state.location.sync(self.table_widget, {'selection': 'selected'})

        # Ensure table reflects current shared selection
        self._apply_table_selection(
            [str(record_id) for record_id in self.data_holder.selected_record_ids]
        )

        return self.table_widget
