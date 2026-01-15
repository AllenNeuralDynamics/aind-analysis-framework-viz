"""
Data table component using Panel Tabulator.

Provides a configurable table display with:
- Column selection
- Row selection with callback
- Header filters
- Grouping support
"""

import logging
from typing import Callable, List, Optional

import pandas as pd
import panel as pn
import param

logger = logging.getLogger(__name__)


class DataTable(param.Parameterized):
    """
    Wrapper around Panel Tabulator for displaying DataFrames.

    Features:
    - Dynamic column selection
    - Row selection with callbacks
    - Header-based filtering
    - Optional grouping
    """

    # Additional columns to display beyond defaults
    additional_columns = param.List(default=[], doc="Additional columns selected by user")

    def __init__(
        self,
        df: pd.DataFrame,
        display_columns: Optional[List[str]] = None,
        frozen_columns: Optional[List[str]] = None,
        groupby: Optional[List[str]] = None,
        on_selection: Optional[Callable[[List[str]], None]] = None,
        id_column: str = "_id",
        height: int = 400,
        **params,
    ):
        """
        Initialize the data table.

        Args:
            df: DataFrame to display
            display_columns: Default columns to show
            frozen_columns: Columns to freeze on left
            groupby: Columns to group by
            on_selection: Callback when rows are selected, receives list of record IDs
            id_column: Column name for record identifier
            height: Table height in pixels
        """
        super().__init__(**params)
        self.df = df
        self.default_display_columns = display_columns or list(df.columns)
        self.frozen_columns = frozen_columns or []
        self.groupby = groupby or []
        self.on_selection = on_selection
        self.id_column = id_column
        self.height = height

        self._tabulator: Optional[pn.widgets.Tabulator] = None

    def get_display_columns(self) -> List[str]:
        """Get combined default and additional columns that exist in df."""
        cols = [c for c in self.default_display_columns if c in self.df.columns]
        additional = [c for c in self.additional_columns if c in self.df.columns and c not in cols]
        return cols + additional

    def create_table(self, df: Optional[pd.DataFrame] = None) -> pn.widgets.Tabulator:
        """
        Create and return the Tabulator widget.

        Args:
            df: Optional DataFrame to use (overrides init df)

        Returns:
            Configured Tabulator widget
        """
        data = df if df is not None else self.df

        # Get display columns (defaults + additional)
        cols_to_show = self.get_display_columns()
        display_df = data[cols_to_show] if cols_to_show else data

        self._tabulator = pn.widgets.Tabulator(
            display_df,
            selectable="checkbox",
            disabled=True,
            frozen_columns=[c for c in self.frozen_columns if c in cols_to_show],
            groupby=[c for c in self.groupby if c in cols_to_show] or None,
            header_filters=True,
            show_index=False,
            height=self.height,
            sizing_mode="stretch_width",
            pagination=None,
            stylesheets=[":host .tabulator {font-size: 12px;}"],
        )

        # Set up selection callback
        if self.on_selection:
            def handle_selection(event):
                if event.new:
                    # Get IDs from all selected indices
                    selected_ids = []
                    if self.id_column in data.columns:
                        for selected_index in event.new:
                            record_id = str(data.iloc[selected_index][self.id_column])
                            selected_ids.append(record_id)
                    self.on_selection(selected_ids)
                else:
                    # No selection
                    self.on_selection([])

            self._tabulator.param.watch(handle_selection, "selection")

        return self._tabulator

    def create_column_selector(
        self,
        height: int = 150,
    ) -> pn.Column:
        """
        Create a widget panel for selecting additional columns to display.

        Args:
            height: Selector height in pixels

        Returns:
            Panel with MultiSelect widget and status message
        """
        # Get available columns (excluding defaults)
        default_cols = set(self.default_display_columns)
        available_cols = sorted(
            [col for col in self.df.columns if col not in default_cols],
            key=str.lower,
        )

        # Status message
        if available_cols:
            status_msg = f"*{len(available_cols)} additional columns available*"
        else:
            status_msg = "*Load data to see available columns*"

        # Create multi-select widget
        column_selector = pn.widgets.MultiSelect(
            name="Additional Columns",
            options=available_cols,
            value=self.additional_columns,
            size=min(8, len(available_cols)) if available_cols else 8,
            sizing_mode="stretch_width",
        )

        # Update additional_columns when selection changes
        def on_column_change(event):
            self.additional_columns = list(event.new)

        column_selector.param.watch(on_column_change, "value")

        return pn.Column(
            pn.pane.Markdown("### Additional Columns"),
            pn.pane.Markdown(status_msg),
            column_selector,
            sizing_mode="stretch_width",
        )


def create_reactive_table(
    df_param: param.Parameter,
    display_columns: List[str],
    id_column: str = "_id",
    on_selection: Optional[Callable[[str], None]] = None,
    height: int = 400,
) -> pn.widgets.Tabulator:
    """
    Create a table that reacts to DataFrame parameter changes.

    Args:
        df_param: Parameter containing the DataFrame
        display_columns: Columns to display
        id_column: ID column name
        on_selection: Selection callback
        height: Table height

    Returns:
        Bound Tabulator that updates when df_param changes
    """
    def make_table(df):
        if df is None or df.empty:
            return pn.pane.Markdown("No data available")

        table = DataTable(
            df=df,
            display_columns=display_columns,
            id_column=id_column,
            on_selection=on_selection,
            height=height,
        )
        return table.create_table()

    return pn.bind(make_table, df_param)
