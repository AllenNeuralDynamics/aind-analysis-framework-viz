"""DocDB query panel component for custom MongoDB queries."""

import json
import logging
from typing import TYPE_CHECKING, Callable

import panel as pn

from .base import BaseComponent

if TYPE_CHECKING:
    from config import AppConfig
    from core.base_app import DataHolder

logger = logging.getLogger(__name__)


class DocDBQueryPanel(BaseComponent):
    """
    Component for editing and executing DocDB (MongoDB) queries.

    Provides a JSON editor for custom queries with reload functionality.
    Displayed as a collapsible card in the sidebar.
    """

    def __init__(
        self,
        data_holder: "DataHolder",
        config: "AppConfig",
        load_data_callback: Callable[[dict], str],
        get_default_query: Callable[[], dict],
    ):
        """
        Initialize the DocDB query panel.

        Args:
            data_holder: Shared state container
            config: Current project configuration
            load_data_callback: Function to call when loading data,
                takes query dict and returns status message
            get_default_query: Function to get the default query dict
        """
        super().__init__(data_holder, config)
        self.load_data_callback = load_data_callback
        self.get_default_query = get_default_query
        self.docdb_query_widget = None
        self._url_sync_initialized = False

    def get_current_query(self) -> dict | None:
        """
        Get the current query from the widget as a dict.

        Returns:
            Query dict or None if widget is not initialized or JSON is invalid
        """
        if self.docdb_query_widget is None:
            return None
        try:
            return json.loads(str(self.docdb_query_widget.value))
        except json.JSONDecodeError:
            return None

    def set_query(self, query_dict: dict, update_url: bool = True) -> None:
        """
        Set the query widget value from a dict.

        Args:
            query_dict: Query dict to set
            update_url: If True, also update the URL param (default True).
                        Set to False to avoid sync loops.
        """
        if self.docdb_query_widget is not None:
            # Format with indent=2 for display
            self.docdb_query_widget.value = json.dumps(query_dict, indent=2)

    def create(self) -> pn.Card:
        """Create the DocDB query panel UI."""
        # Get default query - this is the initial value before URL sync
        default_query = json.dumps(self.get_default_query(), indent=2)

        self.docdb_query_widget = pn.widgets.TextAreaInput(
            name="DocDB Query (JSON)",
            value=default_query,
            placeholder='e.g., {"subject_id": "778869"}',
            height=100,
            sizing_mode="stretch_width",
        )

        # Bidirectional sync with URL using location.sync()
        # This syncs the widget value (JSON string) to URL param 'docdb_query'
        if not self._url_sync_initialized:
            location = pn.state.location
            location.sync(self.docdb_query_widget, {"value": "docdb_query"})
            self._url_sync_initialized = True

        reload_button = pn.widgets.Button(
            name="Reload Data",
            button_type="primary",
            width=120,
        )
        status = pn.pane.Markdown("", css_classes=["alert", "alert-info", "p-2"])

        # Loading indicator (hidden by default)
        loading_spinner = pn.indicators.LoadingSpinner(
            value=False,
            width=30,
            height=30,
            sizing_mode="fixed",
        )

        def reload_callback(_event):
            # Show loading spinner and update status
            loading_spinner.value = True
            status.object = "**Loading data...**"

            try:
                query_str = str(self.docdb_query_widget.value)
                query = json.loads(query_str)
                result = self.load_data_callback(query)
                status.object = result
            except json.JSONDecodeError as e:
                status.object = f"Invalid JSON: {e}"
            finally:
                # Hide spinner
                loading_spinner.value = False

        reload_button.on_click(reload_callback)

        # Build example queries from config
        examples = "\n**Example queries:**\n"
        for ex in self.config.query.get_example_queries():
            examples += f"- `{ex}`\n"

        examples_card = pn.Card(
            pn.pane.Markdown(examples),
            title="Example Queries",
            collapsed=True,
        )

        # Wrap the entire panel in a Card, collapsed by default
        query_panel = pn.Card(
            self.docdb_query_widget,
            pn.Row(reload_button, loading_spinner),
            status,
            examples_card,
            title="DocDB Query",
            collapsed=True,
            sizing_mode="stretch_width",
            stylesheets=[""":host { margin: 0 10px 10px 10px; }"""],
        )

        return query_panel
