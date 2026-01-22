"""
Color mapping utilities for scatter plot visualization.

Handles both categorical and continuous color mapping based on data characteristics.
Follows patterns from LCNE-patchseq-viz for robust color handling.
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
    Blues256,
    Cividis256,
    Greens256,
    Greys256,
    Inferno256,
    Magma256,
    Oranges256,
    Plasma256,
    Purples256,
    Reds256,
    Turbo256,
    Viridis256,
    all_palettes,
)
from bokeh.plotting import figure as Figure

# Special values that should be colored gray
GRAY_VALUES = {
    "None",
    "none",
    "unknown",
    "Unknown",
    "N/A",
    "NA",
    "nan",
    "NaN",
    "seq_data_not_available",
    "",
}

# Named palette map for quick access
NAMED_PALETTES = {
    "Category10": Category10,
    "Category20": Category20,
    "Viridis256": Viridis256,
    "Plasma256": Plasma256,
    "Magma256": Magma256,
    "Inferno256": Inferno256,
    "Turbo256": Turbo256,
    "Cividis256": Cividis256,
    "Greys256": Greys256,
    "Blues256": Blues256,
    "Greens256": Greens256,
    "Oranges256": Oranges256,
    "Reds256": Reds256,
    "Purples256": Purples256,
}


def get_palette(palette_name: str) -> list[str]:
    """Get a color palette by name.

    Tries named palettes first, then looks in all_palettes.
    """
    # Try named palettes
    if palette_name in NAMED_PALETTES:
        palette = NAMED_PALETTES[palette_name]
        # If it's a dict of dicts (like all_palettes format), get the largest one
        if isinstance(palette, dict):
            max_key = max(palette.keys())
            return palette[max_key]
        return palette

    # Try all_palettes with the largest available size
    if palette_name in all_palettes:
        palette_dict = all_palettes[palette_name]
        max_key = max(palette_dict.keys())
        return palette_dict[max_key]

    # Fallback to Category10
    return Category10[10]


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}".format(rgb[0]) + "{:02x}".format(rgb[1]) + "{:02x}".format(rgb[2])


def _interpolate_palette(palette: list[str], n_colors: int = 256) -> list[str]:
    if len(palette) >= n_colors:
        return palette[:n_colors]

    rgb = np.array([_hex_to_rgb(color) for color in palette], dtype=float)
    base_positions = np.linspace(0, 1, len(palette))
    target_positions = np.linspace(0, 1, n_colors)
    channels = []
    for channel in range(3):
        channels.append(np.interp(target_positions, base_positions, rgb[:, channel]))
    interp_rgb = np.stack(channels, axis=1).round().astype(int)
    return [_rgb_to_hex(tuple(values)) for values in interp_rgb]


def _select_categorical_palette(palette_name: str, n_categories: int) -> list[str]:
    """Select appropriate categorical palette given number of categories.

    Uses intelligent palette selection:
    - For small n: uses exact palette size from all_palettes
    - For larger n: samples uniformly from continuous palette
    - Falls back to cycling if necessary
    """
    if palette_name in all_palettes:
        palette_dict = all_palettes[palette_name]

        # Find closest available size
        available_sizes = sorted(palette_dict.keys())

        # If we have an exact match, use it
        if n_categories in available_sizes:
            return palette_dict[n_categories]

        # If n_categories is smaller than max, use closest larger size
        # and uniformly sample from it
        larger_sizes = [s for s in available_sizes if s >= n_categories]
        if larger_sizes:
            closest_size = min(larger_sizes)
            palette = palette_dict[closest_size]
            # Uniformly sample n_categories colors across the full palette
            indices = np.linspace(0, len(palette) - 1, n_categories).astype(int)
            return [palette[i] for i in indices]

        # If n_categories is larger than all available, use the largest
        # and sample uniformly to get n_categories colors
        max_size = available_sizes[-1]
        continuous_palette = palette_dict[max_size]
        indices = np.linspace(0, len(continuous_palette) - 1, n_categories).astype(int)
        return [continuous_palette[i] for i in indices]

    # Fallback
    palette = list(get_palette(palette_name))
    # If a long/continuous palette is selected (e.g. "*256"), spread categories
    # across the full range so adjacent categories don't look identical.
    if len(palette) >= 32 and n_categories > 1 and n_categories < len(palette):
        indices = np.linspace(0, len(palette) - 1, n_categories).astype(int)
        return [palette[i] for i in indices]

    return palette


def determine_color_mapping(
    df: pd.DataFrame,
    color_column: str | None,
    palette_name: str = "Category10",
    max_categorical: int = 50,
    reverse: bool = False,
) -> tuple[dict, CategoricalColorMapper | LinearColorMapper | None]:
    """Determine appropriate color mapping based on data characteristics.

    Implements three-tier approach:
    1. Categorical detection (≤50 unique values or object dtype)
    2. Continuous detection (>50 numeric values)
    3. Fallback to single color if no column specified

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

    # Determine if categorical or continuous
    is_categorical = (
        n_unique <= max_categorical or series.dtype == "object" or series.dtype.name == "category"
    )

    if is_categorical:
        return _create_categorical_mapping(series, palette_name, reverse)
    else:
        return _create_continuous_mapping(series, palette_name, reverse)


def _create_categorical_mapping(
    series: pd.Series, palette_name: str, reverse: bool
) -> tuple[dict, CategoricalColorMapper]:
    """Create categorical color mapping with intelligent palette selection.

    Features:
    - Alphabetical factor ordering for reproducibility
    - Intelligent palette sizing
    - Gray coloring for unknown/missing values
    """
    # Get unique values, handling NaN
    unique_values = series.dropna().unique().tolist()

    # Convert to strings for consistent handling
    unique_values = [str(v) for v in unique_values]

    # Sort alphabetically for reproducibility (reverse order like LCNE)
    try:
        factors = sorted(unique_values, reverse=True)
    except TypeError:
        # Mixed types - just use as-is
        factors = sorted(set(unique_values))

    # Select appropriate palette (already uniformly sampled for continuous palettes)
    palette = _select_categorical_palette(palette_name, len(factors))
    if reverse:
        palette = list(reversed(palette))

    # Assign colors to factors, overriding gray for special values
    colors = []
    palette_idx = 0
    for factor in factors:
        if factor in GRAY_VALUES:
            colors.append("#808080")  # Gray for unknown/missing values
        else:
            colors.append(palette[palette_idx % len(palette)])
            palette_idx += 1

    mapper = CategoricalColorMapper(
        factors=factors,
        palette=colors,
        nan_color="#808080",  # Gray for NaN
    )

    return {"field": series.name, "transform": mapper}, mapper


def _create_continuous_mapping(
    series: pd.Series, palette_name: str, reverse: bool
) -> tuple[dict, LinearColorMapper]:
    """Create continuous color mapping using percentile range.

    Uses 1st-99th percentile bounds to avoid outlier-driven scaling.
    """
    # Convert to numeric, coercing errors to NaN
    numeric_series = pd.to_numeric(series, errors="coerce")

    # Use 1st-99th percentile range to avoid outlier influence
    low = float(numeric_series.quantile(0.01))
    high = float(numeric_series.quantile(0.99))

    # Handle case where all values are the same
    if low == high:
        low = float(numeric_series.min())
        high = float(numeric_series.max())
        if low == high:
            high = low + 1  # Prevent division by zero in LinearColorMapper

    # Get continuous palette - use largest available if palette_name is in all_palettes
    if palette_name in all_palettes:
        palette_dict = all_palettes[palette_name]
        max_key = max(palette_dict.keys())
        base_palette = palette_dict[max_key]
        continuous_palette = _interpolate_palette(base_palette, 256)
    else:
        continuous_palette = _interpolate_palette(get_palette(palette_name), 256)

    if reverse:
        continuous_palette = list(reversed(continuous_palette))

    mapper = LinearColorMapper(
        palette=continuous_palette,
        low=low,
        high=high,
        nan_color="#808080",  # Gray for NaN
    )

    return {"field": series.name, "transform": mapper}, mapper


def add_color_bar(
    fig: Figure,
    mapper: LinearColorMapper | CategoricalColorMapper | None,
    title: str = "",
    font_size: int = 12,
) -> None:
    """
    Add a color bar to the figure for both categorical and continuous mappings.

    Bokeh's ColorBar handles both mapper types:
    - CategoricalColorMapper: Shows discrete color blocks with labels
    - LinearColorMapper: Shows continuous gradient

    Args:
        fig: Bokeh figure to add color bar to
        mapper: Color mapper (categorical or continuous)
        title: Title for the color bar
        font_size: Font size for color bar labels
    """
    if mapper is None:
        return

    color_bar = ColorBar(
        color_mapper=mapper,
        title=title,
        label_standoff=12,
        border_line_color=None,
        location=(0, 0),
        title_text_font_size=f"{int(font_size * 0.9)}pt",
        major_label_text_font_size=f"{int(font_size * 0.8)}pt",
    )
    fig.add_layout(color_bar, "right")
