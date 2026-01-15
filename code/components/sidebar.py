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
        # Selected count
        selected_display = pn.bind(
            lambda ids: pn.pane.Markdown(
                f"**Count:** {len(ids)}" if ids else "**Count:** 0",
                css_classes=["alert", "alert-secondary", "p-2"],
            ),
            ids=self.data_holder.param.selected_record_ids,
        )

        # Records count
        records_display = pn.bind(
            lambda df: pn.pane.Markdown(
                f"**Records:** {len(df) if df is not None else 0}",
                css_classes=["alert", "alert-info", "p-2"],
            ),
            df=self.data_holder.param.filtered_df,
        )

        # Subjects count (if subject_id_column is configured)
        def render_subjects(df):
            if (
                df is not None
                and self.config
                and self.config.subject_id_column
                and self.config.subject_id_column in df.columns
            ):
                count = df[self.config.subject_id_column].nunique()
                return pn.pane.Markdown(
                    f"**Subjects:** {count}",
                    css_classes=["alert", "alert-info", "p-2"],
                )
            return pn.pane.Markdown(
                "**Subjects:** N/A",
                css_classes=["alert", "alert-info", "p-2"],
            )

        subjects_display = pn.bind(
            render_subjects,
            df=self.data_holder.param.filtered_df,
        )

        return pn.Column(
            pn.pane.Markdown("### Selected"),
            selected_display,
            pn.pane.Markdown("### Stats"),
            records_display,
            subjects_display,
            sizing_mode="stretch_width",
        )
