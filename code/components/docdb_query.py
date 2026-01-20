"""DocDB query panel component for custom MongoDB queries."""

import json
import logging
from typing import TYPE_CHECKING, Callable

import panel as pn

from .base import BaseComponent
from utils import get_url_param, update_url_param

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

    def set_query(self, query_dict: dict) -> None:
        """
        Set the query widget value from a dict.

        Args:
            query_dict: Query dict to set
        """
        if self.docdb_query_widget is not None:
            self.docdb_query_widget.value = json.dumps(query_dict, indent=2)

    def create(self) -> pn.Card:
        """Create the DocDB query panel UI."""
        default_query = json.dumps(self.get_default_query(), indent=2)

        docdb_query = pn.widgets.TextAreaInput(
            name="DocDB Query (JSON)",
            value=default_query,
            placeholder='e.g., {"subject_id": "778869"}',
            height=100,
            sizing_mode="stretch_width",
        )

        self.docdb_query_widget = docdb_query

        # Read docdb_query from URL on initial load
        url_query = get_url_param("docdb_query")
        if url_query:
            try:
                query_dict = json.loads(url_query)
                docdb_query.value = json.dumps(query_dict, indent=2)
                logger.info(f"Loaded DocDB query from URL: {url_query}")
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid docdb_query in URL: {e}")

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
                query_str = str(docdb_query.value)
                query = json.loads(query_str)
                result = self.load_data_callback(query)
                status.object = result

                # Update URL with the query on successful reload
                update_url_param("docdb_query", json.dumps(query, separators=(",", ":")))
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
            docdb_query,
            pn.Row(reload_button, loading_spinner),
            status,
            examples_card,
            title="DocDB Query",
            collapsed=True,
            sizing_mode="stretch_width",
            stylesheets=[""":host { margin: 0 10px 10px 10px; }"""],
        )

        return query_panel
