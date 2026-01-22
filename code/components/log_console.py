"""Sidebar console component for log and print output."""

from __future__ import annotations

import logging
import sys

import panel as pn


class LogConsole:
    """Read-only console that mirrors logging + print output."""

    def __init__(self, max_lines: int = 200) -> None:
        self._log_lines: list[str] = []
        self._max_log_lines = max_lines
        self._console_handler: logging.Handler | None = None
        self._stdout = sys.stdout
        self._enabled = False
        self._console_writer: object | None = None

        self.toggle = pn.widgets.Toggle(
            name="Show debugging info",
            value=False,
            button_type="warning",
            width=200,
        )
        self.toggle.param.watch(self._on_toggle, "value")

        self.widget = pn.widgets.TextAreaInput(
            name="Debugging console",
            value="",
            height=400,
            sizing_mode="stretch_width",
            disabled=True,
            resizable="height",
            styles={"font-size": "11px"},
        )
        self.widget.visible = False
        self._setup_logging()

    def create(self) -> pn.Column:
        location = pn.state.location
        if location is not None:
            location.sync(self.toggle, {"value": "debug"})

        return pn.Column(
            self.toggle,
            self.widget,
            sizing_mode="stretch_width",
        )

    def _append_line(self, message: str) -> None:
        if not message:
            return
        self._log_lines.append(message)
        if len(self._log_lines) > self._max_log_lines:
            self._log_lines = self._log_lines[-self._max_log_lines :]
        self.widget.value = "\n".join(self._log_lines)

    def _setup_logging(self) -> None:
        class ConsoleHandler(logging.Handler):
            def emit(handler_self, record) -> None:
                try:
                    message = handler_self.format(record)
                except Exception:
                    message = str(record.getMessage())
                self._append_line(message)

        class ConsoleWriter:
            def write(writer_self, text: str) -> int:
                if not text:
                    return 0
                lines = text.rstrip().splitlines()
                for line in lines:
                    self._append_line(line)
                return len(text)

            def flush(writer_self) -> None:
                return None

        self._console_handler = ConsoleHandler()
        self._console_handler.setFormatter(
            logging.Formatter("[%(levelname)s] %(message)s")
        )
        self._console_writer = ConsoleWriter()

    def _on_toggle(self, event) -> None:
        if event.new:
            self._enable()
        else:
            self._disable()

    def _enable(self) -> None:
        if self._enabled:
            return
        root_logger = logging.getLogger()
        if self._console_handler and self._console_handler not in root_logger.handlers:
            root_logger.addHandler(self._console_handler)
        if self._console_writer and sys.stdout is not self._console_writer:
            self._stdout = sys.stdout
            sys.stdout = self._console_writer
        self.widget.visible = True
        self._enabled = True

    def _disable(self) -> None:
        if not self._enabled:
            return
        root_logger = logging.getLogger()
        if self._console_handler and self._console_handler in root_logger.handlers:
            root_logger.removeHandler(self._console_handler)
        if sys.stdout is self._console_writer:
            sys.stdout = self._stdout
        self.widget.visible = False
        self._enabled = False
