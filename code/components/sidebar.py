"""Sidebar components for project selection, data loading, and stats."""

import logging
from typing import TYPE_CHECKING, Callable

import panel as pn

from .base import BaseComponent

if TYPE_CHECKING:
    from config import AppConfig
    from core.base_app import DataHolder

logger = logging.getLogger(__name__)


class LoadDataPanel(BaseComponent):
    """
    Component for the Load Data button with status and loading indicator.
    """

    def __init__(
        self,
        data_holder: "DataHolder",
        config: "AppConfig",
        load_data_callback: Callable[[], str],
    ):
        """
        Initialize the load data panel.

        Args:
            data_holder: Shared state container
            config: Current project configuration
            load_data_callback: Function to call when loading data,
                returns status message
        """
        super().__init__(data_holder, config)
        self.load_data_callback = load_data_callback

    def create(self) -> pn.Column:
        """Create the load data panel UI."""
        load_button = pn.widgets.Button(
            name="Load Data from DocDB",
            button_type="primary",
            sizing_mode="stretch_width",
        )
        status = pn.pane.Markdown("", css_classes=["alert", "alert-info", "p-2"])

        # Loading indicator (hidden by default)
        loading_spinner = pn.indicators.LoadingSpinner(
            value=False,
            width=30,
            height=30,
            sizing_mode="fixed",
        )

        def load_callback(_event):
            # Show loading spinner and update status
            loading_spinner.value = True
            status.object = "**Loading data...**"

            # Load the data
            result = self.load_data_callback()

            # Hide spinner and show result
            loading_spinner.value = False
            status.object = result

        load_button.on_click(load_callback)

        return pn.Column(
            pn.Row(load_button, loading_spinner),
            status,
            sizing_mode="stretch_width",
        )


class StatsPanel(BaseComponent):
    """
    Component for displaying statistics about the current data.

    Shows record count, subject count, and selection count.
    """

    def create(self) -> pn.Column:
        """Create the stats panel UI with reactive bindings."""

        def render_stats(ids, df):
            """Render combined stats with filtered and selected info."""
            if df is None or df.empty:
                return pn.pane.Markdown(
                    "**Filtered:** 0 records, 0 subjects  \n**Selected:** 0 records, 0 subjects",
                    css_classes=["alert", "alert-secondary", "p-1"],
                    styles={"font-size": "12px"},
                )

            # Get filtered stats
            n_records = len(df)
            if (
                self.config
                and self.config.subject_id_column
                and self.config.subject_id_column in df.columns
            ):
                n_subjects = df[self.config.subject_id_column].nunique()
            else:
                n_subjects = 0

            # Get selected stats
            n_selected = len(ids)
            if n_selected > 0 and self.config.subject_id_column:
                # Filter df to selected records and count subjects
                df_selected = df[df[self.config.id_column].astype(str).isin(ids)]
                n_selected_subjects = df_selected[self.config.subject_id_column].nunique()
            else:
                n_selected_subjects = 0

            return pn.pane.Markdown(
                f"**Filtered:** {n_records} records, {n_subjects} subjects  \n"
                f"**Selected:** {n_selected} records, {n_selected_subjects} subjects",
                css_classes=["alert", "alert-secondary", "p-1"],
                styles={"font-size": "12px"},
            )

        stats_display = pn.bind(
            render_stats,
            ids=self.data_holder.param.selected_record_ids,
            df=self.data_holder.param.filtered_df,
        )

        return pn.Column(
            stats_display,
            sizing_mode="stretch_width",
            css_classes=["p-2"],
        )
