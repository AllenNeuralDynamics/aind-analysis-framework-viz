"""Base component protocol for UI components.

All UI components should follow this pattern for consistency.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import panel as pn

if TYPE_CHECKING:
    from config import AppConfig
    from core.base_app import DataHolder


class BaseComponent(ABC):
    """
    Abstract base class for UI components.

    Components receive the shared DataHolder for reactive state management
    and the current AppConfig for project-specific settings.

    Usage:
        class MyComponent(BaseComponent):
            def create(self) -> pn.viewable.Viewable:
                return pn.bind(
                    self._render,
                    df=self.data_holder.param.filtered_df,
                )
    """

    def __init__(self, data_holder: "DataHolder", config: "AppConfig"):
        """
        Initialize the component.

        Args:
            data_holder: Shared state container for reactive updates
            config: Current project configuration
        """
        self.data_holder = data_holder
        self.config = config

    @abstractmethod
    def create(self) -> pn.viewable.Viewable:
        """
        Create and return the Panel component.

        Returns:
            A Panel viewable object (Column, Row, pane, widget, etc.)
        """
        pass
