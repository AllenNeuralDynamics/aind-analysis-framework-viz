"""
Color mapping utilities for scatter plot visualization.

Handles both categorical and continuous color mapping based on data characteristics.
"""

import numpy as np
import pandas as pd
from bokeh.models import (
    CategoricalColorMapper,
    ColorBar,
    LinearColorMapper,
)
from bokeh.palettes import (
    Category10,
    Category20,
    Cividis256,
    Magma256,
    Plasma256,
    Turbo256,
    Viridis256,
)
from bokeh.plotting import figure as Figure

# Palette name to actual palette mapping
PALETTE_MAP = {
    "Category10": Category10[10],
    "Category20": Category20[20],
    "Viridis256": Viridis256,
    "Plasma256": Plasma256,
    "Magma256": Magma256,
    "Turbo256": Turbo256,
    "Cividis256": Cividis256,
}

# Special values that should be colored gray
GRAY_VALUES = {"None", "none", "unknown", "Unknown", "N/A", "NA", "nan", "NaN", ""}


def get_palette(palette_name: str) -> list[str]:
    """Get a color palette by name."""
    return PALETTE_MAP.get(palette_name, Category10[10])


def determine_color_mapping(
    df: pd.DataFrame,
    color_column: str | None,
    palette_name: str = "Category10",
    max_categorical: int = 50,
) -> tuple[dict, CategoricalColorMapper | LinearColorMapper | None]:
    """
    Determine appropriate color mapping based on data characteristics.

    Args:
        df: DataFrame containing the data
        color_column: Column name to use for coloring (None for no coloring)
        palette_name: Name of the color palette to use
        max_categorical: Maximum unique values for categorical treatment

    Returns:
        Tuple of (color_spec_dict, color_mapper):
        - color_spec_dict: Dict with 'field' and 'transform' keys for Bokeh glyph
        - color_mapper: The ColorMapper instance (or None if no color column)
    """
    if not color_column or color_column not in df.columns:
        # No color mapping - return default color
        return {"value": "#1f77b4"}, None

    series = df[color_column].copy()
    n_unique = series.nunique()
    palette = get_palette(palette_name)

    # Determine if categorical or continuous
    is_categorical = (
        n_unique <= max_categorical
        or series.dtype == "object"
        or series.dtype.name == "category"
    )

    if is_categorical:
        return _create_categorical_mapping(series, palette)
    else:
        return _create_continuous_mapping(series, palette_name)


def _create_categorical_mapping(
    series: pd.Series, palette: list[str]
) -> tuple[dict, CategoricalColorMapper]:
    """Create categorical color mapping."""
    # Get unique values, handling NaN
    unique_values = series.dropna().unique().tolist()

    # Sort for consistent ordering
    try:
        unique_values = sorted(unique_values)
    except TypeError:
        # Mixed types - convert to string
        unique_values = sorted([str(v) for v in unique_values])

    # Create color mapping
    n_colors = len(unique_values)
    if n_colors <= len(palette):
        colors = palette[:n_colors]
    else:
        # Cycle through palette if more categories than colors
        colors = [palette[i % len(palette)] for i in range(n_colors)]

    # Override gray values
    for i, val in enumerate(unique_values):
        if str(val) in GRAY_VALUES:
            colors[i] = "#808080"  # Gray

    mapper = CategoricalColorMapper(
        factors=[str(v) for v in unique_values],
        palette=colors,
        nan_color="#808080",
    )

    return {"field": series.name, "transform": mapper}, mapper


def _create_continuous_mapping(
    series: pd.Series, palette_name: str
) -> tuple[dict, LinearColorMapper]:
    """Create continuous color mapping using percentile range."""
    # Convert to numeric, coercing errors
    numeric_series = pd.to_numeric(series, errors="coerce")

    # Use 1st-99th percentile range to avoid outlier influence
    low = float(numeric_series.quantile(0.01))
    high = float(numeric_series.quantile(0.99))

    # Get continuous palette
    if palette_name in ["Category10", "Category20"]:
        # These are categorical - use Viridis for continuous
        continuous_palette = Viridis256
    else:
        continuous_palette = get_palette(palette_name)

    mapper = LinearColorMapper(
        palette=continuous_palette,
        low=low,
        high=high,
        nan_color="#808080",
    )

    return {"field": series.name, "transform": mapper}, mapper


def add_color_bar(
    fig: Figure,
    mapper: LinearColorMapper | CategoricalColorMapper | None,
    title: str = "",
) -> None:
    """
    Add a color bar to the figure if mapper is continuous.

    Args:
        fig: Bokeh figure to add color bar to
        mapper: Color mapper (only LinearColorMapper will add a bar)
        title: Title for the color bar
    """
    if mapper is None or isinstance(mapper, CategoricalColorMapper):
        return

    color_bar = ColorBar(
        color_mapper=mapper,
        title=title,
        label_standoff=12,
        border_line_color=None,
        location=(0, 0),
    )
    fig.add_layout(color_bar, "right")
