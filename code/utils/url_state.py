"""URL state utilities for one-way sync pattern.

Panel's location.sync() creates bidirectional bindings that can cause
race conditions when multiple widgets update the URL. This module provides
one-way sync helpers: read from URL once on init, write to URL on changes.
"""

import ast
import logging
from typing import Any, List, Optional
from urllib.parse import parse_qs

import panel as pn

logger = logging.getLogger(__name__)


def get_url_param(param_name: str, default: Any = None) -> Optional[str]:
    """Read a single URL parameter value.

    Args:
        param_name: URL parameter name
        default: Default value if param not found

    Returns:
        Parameter value or default
    """
    query_string = pn.state.location.search or ""
    if query_string.startswith("?"):
        query_string = query_string[1:]
    params = parse_qs(query_string)
    values = params.get(param_name, [])
    return values[0] if values else default


def get_url_param_list(param_name: str) -> List[str]:
    """Read a list parameter from URL.

    Handles both formats:
    - "['a','b','c']" (Python list literal)
    - "a,b,c" (comma-separated)

    Args:
        param_name: URL parameter name

    Returns:
        List of values (empty list if not found)
    """
    value = get_url_param(param_name)
    if not value:
        return []

    # Try Python list literal format first
    if value.startswith("["):
        try:
            result = ast.literal_eval(value)
            return result if isinstance(result, list) else []
        except (ValueError, SyntaxError):
            return []

    # Fall back to comma-separated
    return [v.strip() for v in value.split(",") if v.strip()]


def update_url_param(param_name: str, value: Any) -> None:
    """Update a single URL parameter.

    Args:
        param_name: URL parameter name
        value: New value (None to remove param)
    """
    pn.state.location.update_query(**{param_name: value})
