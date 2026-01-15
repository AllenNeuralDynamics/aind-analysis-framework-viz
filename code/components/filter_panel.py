"""Filter panel component for pandas query filtering."""

import logging
from typing import TYPE_CHECKING, Callable

import panel as pn

from .base import BaseComponent

if TYPE_CHECKING:
    from config import AppConfig
    from core.base_app import DataHolder

logger = logging.getLogger(__name__)


class FilterPanel(BaseComponent):
    """
    Component for filtering data using pandas query syntax.

    Provides a text input for query strings with apply/reset buttons
    and example queries from the config.
    """

    def __init__(
        self,
        data_holder: "DataHolder",
        config: "AppConfig",
        apply_filter_callback: Callable[[str], str],
    ):
        """
        Initialize the filter panel.

        Args:
            data_holder: Shared state container
            config: Current project configuration
            apply_filter_callback: Function to call when applying filter,
                takes query string and returns status message
        """
        super().__init__(data_holder, config)
        self.apply_filter_callback = apply_filter_callback

    def create(self) -> pn.Column:
        """Create the filter panel UI."""
        filter_query = pn.widgets.TextAreaInput(
            name="Pandas Query",
            value="",
            placeholder=self.config.filter.default_placeholder if self.config else "",
            height=60,
            sizing_mode="stretch_width",
        )

        filter_button = pn.widgets.Button(
            name="Apply Filter",
            button_type="primary",
            width=120,
        )
        reset_button = pn.widgets.Button(
            name="Reset",
            button_type="light",
            width=80,
        )
        status = pn.pane.Markdown("", css_classes=["alert", "alert-info", "p-2"])

        def apply_callback(_event):
            result = self.apply_filter_callback(filter_query.value)
            status.object = result

        def reset_callback(_event):
            filter_query.value = ""
            result = self.apply_filter_callback("")
            status.object = result

        filter_button.on_click(apply_callback)
        reset_button.on_click(reset_callback)

        # Build example queries from config
        examples = "\n**Example queries:**\n"
        for ex in self.config.filter.example_queries:
            examples += f"- `{ex}`\n"

        examples_card = pn.Card(
            pn.pane.Markdown(examples),
            title="Example Queries",
            collapsed=True,
        )

        return pn.Column(
            pn.pane.Markdown("### Global Filter"),
            filter_query,
            pn.Row(filter_button, reset_button),
            status,
            examples_card,
            sizing_mode="stretch_width",
        )
