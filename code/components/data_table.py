"""
Data table component using Panel Tabulator.

Provides a configurable table display with:
- Column selection
- Row selection with callback
- Header filters
- Grouping support
"""

import logging
from typing import Any, Callable, List, Optional

import pandas as pd
import panel as pn

logger = logging.getLogger(__name__)


class DataTable:
    """
    Wrapper around Panel Tabulator for displaying DataFrames.

    Features:
    - Dynamic column selection
    - Row selection with callbacks
    - Header-based filtering
    - Optional grouping
    """

    def __init__(
        self,
        df: pd.DataFrame,
        display_columns: Optional[List[str]] = None,
        frozen_columns: Optional[List[str]] = None,
        groupby: Optional[List[str]] = None,
        on_selection: Optional[Callable[[str], None]] = None,
        id_column: str = "_id",
        height: int = 400,
    ):
        """
        Initialize the data table.

        Args:
            df: DataFrame to display
            display_columns: Columns to show (default: all)
            frozen_columns: Columns to freeze on left
            groupby: Columns to group by
            on_selection: Callback when row is selected, receives record ID
            id_column: Column name for record identifier
            height: Table height in pixels
        """
        self.df = df
        self.display_columns = display_columns or list(df.columns)
        self.frozen_columns = frozen_columns or []
        self.groupby = groupby or []
        self.on_selection = on_selection
        self.id_column = id_column
        self.height = height

        self._tabulator: Optional[pn.widgets.Tabulator] = None

    def create_table(self, df: Optional[pd.DataFrame] = None) -> pn.widgets.Tabulator:
        """
        Create and return the Tabulator widget.

        Args:
            df: Optional DataFrame to use (overrides init df)

        Returns:
            Configured Tabulator widget
        """
        data = df if df is not None else self.df

        # Filter to display columns that exist
        cols_to_show = [c for c in self.display_columns if c in data.columns]
        display_df = data[cols_to_show] if cols_to_show else data

        self._tabulator = pn.widgets.Tabulator(
            display_df,
            selectable=1,
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
                    selected_index = event.new[0]
                    # Get the ID from the original (possibly filtered) dataframe
                    if self.id_column in data.columns:
                        record_id = str(data.iloc[selected_index][self.id_column])
                        self.on_selection(record_id)

            self._tabulator.param.watch(handle_selection, "selection")

        return self._tabulator

    def create_column_selector(
        self,
        all_columns: Optional[List[str]] = None,
        default_extra: Optional[List[str]] = None,
        height: int = 200,
        width: int = 300,
    ) -> pn.widgets.MultiSelect:
        """
        Create a widget for selecting additional columns to display.

        Args:
            all_columns: All available columns
            default_extra: Default extra columns to show
            height: Selector height
            width: Selector width

        Returns:
            MultiSelect widget for column selection
        """
        available = all_columns or list(self.df.columns)
        selectable = [c for c in available if c not in self.display_columns]

        return pn.widgets.MultiSelect(
            name="Additional columns",
            options=sorted(selectable),
            value=default_extra or [],
            height=height,
            width=width,
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
