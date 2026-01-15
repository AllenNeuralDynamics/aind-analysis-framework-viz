"""
Size mapping utilities for scatter plot visualization.

Handles numeric-based size mapping with gamma correction.
"""

import numpy as np
import pandas as pd


def determine_size_mapping(
    df: pd.DataFrame,
    size_column: str | None,
    min_size: int = 5,
    max_size: int = 40,
    gamma: float = 1.0,
    default_size: int = 10,
) -> np.ndarray:
    """
    Determine point sizes based on a numeric column.

    Args:
        df: DataFrame containing the data
        size_column: Column name to use for sizing (None for uniform size)
        min_size: Minimum point size in pixels
        max_size: Maximum point size in pixels
        gamma: Gamma correction factor (>1 emphasizes large values, <1 emphasizes small)
        default_size: Default size when no size column is specified

    Returns:
        NumPy array of size values for each data point
    """
    if not size_column or size_column not in df.columns:
        # No size mapping - return uniform size
        return np.full(len(df), default_size, dtype=float)

    series = df[size_column].copy()

    # Convert to numeric, coercing errors to NaN
    numeric_series = pd.to_numeric(series, errors="coerce")

    # Get data range (using full range, not percentiles for size)
    data_min = float(numeric_series.min())
    data_max = float(numeric_series.max())

    # Handle case where all values are the same
    if data_max == data_min:
        return np.full(len(df), (min_size + max_size) / 2, dtype=float)

    # Normalize to [0, 1]
    normalized = (numeric_series - data_min) / (data_max - data_min)

    # Apply gamma correction
    # gamma > 1: emphasizes larger values (makes small values smaller)
    # gamma < 1: emphasizes smaller values (makes small values larger)
    normalized = np.clip(normalized, 0, 1)
    gamma_corrected = np.power(normalized, gamma)

    # Scale to size range
    sizes = min_size + gamma_corrected * (max_size - min_size)

    # Handle NaN values - use minimum size
    sizes = np.where(np.isnan(sizes), min_size, sizes)

    return sizes.astype(float)
