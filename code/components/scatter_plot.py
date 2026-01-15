"""
Interactive scatter plot component for data exploration.

Provides:
- Configurable X/Y axes with dropdown selectors
- Color mapping (categorical/continuous)
- Size mapping with gamma correction
- Hover tooltips with S3 image preview
- Point selection syncing to DataHolder
"""

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import panel as pn
from bokeh.models import ColumnDataSource, HoverTool, TapTool
from bokeh.plotting import figure

from .base import BaseComponent
from .color_mapping import add_color_bar, determine_color_mapping
from .size_mapping import determine_size_mapping

if TYPE_CHECKING:
    from config import AppConfig
    from core.base_app import DataHolder

logger = logging.getLogger(__name__)


class ScatterPlot(BaseComponent):
    """
    Interactive scatter plot component with configurable axes and styling.

    Features:
    - Dropdown selectors for X, Y, Color, and Size columns
    - Automatic categorical/continuous color mapping
    - Size mapping with gamma correction
    - Hover tooltips with embedded S3 images
    - Point tap selection syncing to DataHolder
    """

    def __init__(self, data_holder: "DataHolder", config: "AppConfig"):
        """
        Initialize the scatter plot component.

        Args:
            data_holder: Shared state container for reactive updates
            config: Current project configuration
        """
        super().__init__(data_holder, config)
        self._init_controls()
        self._latest_figure = None
        self._source = None

    def _init_controls(self) -> None:
        """Initialize control widgets for the scatter plot."""
        scatter_config = self.config.scatter_plot

        # Column selectors
        self.x_select = pn.widgets.Select(
            name="X Axis",
            options=[],
            value=None,
            width=180,
        )
        self.y_select = pn.widgets.Select(
            name="Y Axis",
            options=[],
            value=None,
            width=180,
        )
        self.color_select = pn.widgets.Select(
            name="Color By",
            options=["None"],
            value="None",
            width=180,
        )
        self.size_select = pn.widgets.Select(
            name="Size By",
            options=["None"],
            value="None",
            width=180,
        )

        # Color palette selector
        self.palette_select = pn.widgets.Select(
            name="Color Palette",
            options=scatter_config.color_palettes,
            value=scatter_config.color_palettes[0],
            width=180,
        )

        # Alpha slider
        self.alpha_slider = pn.widgets.FloatSlider(
            name="Opacity",
            start=0.1,
            end=1.0,
            step=0.1,
            value=scatter_config.default_alpha,
            width=180,
        )

    def _get_numeric_columns(self, df: pd.DataFrame) -> list[str]:
        """Get list of numeric columns from DataFrame."""
        return df.select_dtypes(include=[np.number]).columns.tolist()

    def _get_all_columns(self, df: pd.DataFrame) -> list[str]:
        """Get list of all columns from DataFrame."""
        return df.columns.tolist()

    def _update_column_options(self, df: pd.DataFrame) -> None:
        """Update dropdown options based on available columns."""
        if df is None or df.empty:
            return

        numeric_cols = self._get_numeric_columns(df)
        all_cols = self._get_all_columns(df)

        # Update X/Y selectors with numeric columns
        self.x_select.options = numeric_cols
        self.y_select.options = numeric_cols

        # Update Color/Size with all columns (plus "None" option)
        self.color_select.options = ["None"] + all_cols
        self.size_select.options = ["None"] + numeric_cols

        # Set defaults from config if available
        scatter_config = self.config.scatter_plot

        if scatter_config.x_column and scatter_config.x_column in numeric_cols:
            self.x_select.value = scatter_config.x_column
        elif numeric_cols:
            self.x_select.value = numeric_cols[0]

        if scatter_config.y_column and scatter_config.y_column in numeric_cols:
            self.y_select.value = scatter_config.y_column
        elif len(numeric_cols) > 1:
            self.y_select.value = numeric_cols[1]
        elif numeric_cols:
            self.y_select.value = numeric_cols[0]

        if scatter_config.color_column and scatter_config.color_column in all_cols:
            self.color_select.value = scatter_config.color_column

        if scatter_config.size_column and scatter_config.size_column in numeric_cols:
            self.size_select.value = scatter_config.size_column

    def _build_tooltip_html(self, df: pd.DataFrame) -> str:
        """Build HTML template for hover tooltip."""
        scatter_config = self.config.scatter_plot
        id_col = self.config.id_column

        # Basic info tooltip
        tooltip_parts = [
            '<div style="max-width: 600px; padding: 10px; background: white; '
            'border: 1px solid #ccc; border-radius: 4px;">',
            f'<div style="font-weight: bold; margin-bottom: 5px;">@{{{id_col}}}</div>',
            '<div style="font-size: 12px; color: #666;">',
            "X: @{x_col}{0.000} | Y: @{y_col}{0.000}",
            "</div>",
        ]

        # Add color/size info if set
        tooltip_parts.append(
            '<div style="font-size: 12px; color: #666; margin-top: 3px;">'
            "Color: @{color_col} | Size: @{size_col}"
            "</div>"
        )

        # Add image if tooltip_image_column is configured
        if scatter_config.tooltip_image_column:
            tooltip_parts.append(
                '<div style="margin-top: 10px;">'
                f'<img src="@{{tooltip_image_url}}{{safe}}" '
                'style="max-width: 500px; height: auto;" '
                'onerror="this.style.display=\'none\'">'
                "</div>"
            )

        tooltip_parts.append("</div>")
        return "".join(tooltip_parts)

    def _create_figure(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        color_col: str | None,
        size_col: str | None,
        palette: str,
        alpha: float,
    ):
        """Create the Bokeh scatter plot figure."""
        scatter_config = self.config.scatter_plot

        if df is None or df.empty or not x_col or not y_col:
            return pn.pane.Markdown(
                "No data available or axes not selected.",
                css_classes=["alert", "alert-info", "p-3"],
            )

        # Filter to rows with valid x/y values
        df_plot = df[[x_col, y_col]].dropna()
        if df_plot.empty:
            return pn.pane.Markdown(
                f"No valid data for selected axes ({x_col}, {y_col}).",
                css_classes=["alert", "alert-warning", "p-3"],
            )

        # Get the full rows for valid x/y
        valid_idx = df_plot.index
        df_valid = df.loc[valid_idx].copy()

        # Determine color mapping
        color_column = color_col if color_col != "None" else None
        color_spec, color_mapper = determine_color_mapping(
            df_valid,
            color_column,
            palette,
            scatter_config.max_categorical_values,
        )

        # Determine size mapping
        size_column = size_col if size_col != "None" else None
        sizes = determine_size_mapping(
            df_valid,
            size_column,
            scatter_config.min_size,
            scatter_config.max_size,
            scatter_config.default_gamma,
            scatter_config.default_size,
        )

        # Prepare data for ColumnDataSource
        source_data = {
            "x": df_valid[x_col].values,
            "y": df_valid[y_col].values,
            "size": sizes,
            "index": df_valid.index.tolist(),
            self.config.id_column: df_valid[self.config.id_column].astype(str).values,
            "x_col": [x_col] * len(df_valid),
            "y_col": [y_col] * len(df_valid),
            "color_col": (
                df_valid[color_column].astype(str).values
                if color_column
                else ["N/A"] * len(df_valid)
            ),
            "size_col": (
                df_valid[size_column].astype(str).values
                if size_column
                else ["N/A"] * len(df_valid)
            ),
        }

        # Add color column data if categorical
        if color_column and color_column in df_valid.columns:
            source_data[color_column] = df_valid[color_column].astype(str).values

        # Add tooltip image URL if configured
        if scatter_config.tooltip_image_column:
            if scatter_config.tooltip_image_column in df_valid.columns:
                # Construct URLs from S3 paths
                from .asset_viewer import get_s3_image_url

                asset_config = self.config.asset
                urls = []
                for _, row in df_valid.iterrows():
                    s3_loc = row.get(scatter_config.tooltip_image_column, "")
                    if s3_loc:
                        base = s3_loc if s3_loc.endswith("/") else f"{s3_loc}/"
                        full_path = f"{base}{asset_config.asset_filename}"
                        url = get_s3_image_url(full_path) or ""
                    else:
                        url = ""
                    urls.append(url)
                source_data["tooltip_image_url"] = urls
            else:
                source_data["tooltip_image_url"] = [""] * len(df_valid)

        self._source = ColumnDataSource(data=source_data)

        # Create figure
        p = figure(
            title=f"{y_col} vs {x_col}",
            x_axis_label=x_col,
            y_axis_label=y_col,
            width=scatter_config.width,
            height=scatter_config.height,
            tools="pan,wheel_zoom,box_zoom,reset",
            active_drag="box_zoom",
        )

        # Add scatter points
        scatter_renderer = p.scatter(
            x="x",
            y="y",
            source=self._source,
            size="size",
            alpha=alpha,
            **color_spec,
        )

        # Add hover tool
        hover = HoverTool(
            tooltips=self._build_tooltip_html(df_valid),
            renderers=[scatter_renderer],
            attachment="right",
        )
        p.add_tools(hover)

        # Add tap tool for selection
        tap = TapTool(renderers=[scatter_renderer])
        p.add_tools(tap)

        # Selection callback
        def on_tap_select(attr, old, new):
            if new:
                selected_idx = new[0]
                record_id = str(self._source.data[self.config.id_column][selected_idx])
                logger.debug(f"Selected record: {record_id}")
                self.data_holder.selected_record_ids = [record_id]

        self._source.selected.on_change("indices", on_tap_select)

        # Add color bar for continuous mappings
        if color_column:
            add_color_bar(p, color_mapper, color_column)

        # Style the plot
        p.title.text_font_size = "14pt"
        p.xaxis.axis_label_text_font_size = "12pt"
        p.yaxis.axis_label_text_font_size = "12pt"

        self._latest_figure = p
        return pn.pane.Bokeh(p, sizing_mode="stretch_width")

    def _render_plot(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        color_col: str,
        size_col: str,
        palette: str,
        alpha: float,
    ):
        """Render the scatter plot with current settings."""
        # Update column options when data changes
        self._update_column_options(df)

        return self._create_figure(
            df, x_col, y_col, color_col, size_col, palette, alpha
        )

    def create(self) -> pn.viewable.Viewable:
        """
        Create the scatter plot component with controls.

        Returns:
            Panel viewable with controls and reactive plot
        """
        # Control panel
        controls = pn.Row(
            self.x_select,
            self.y_select,
            self.color_select,
            self.size_select,
            self.palette_select,
            self.alpha_slider,
            sizing_mode="stretch_width",
        )

        # Reactive plot
        plot = pn.bind(
            self._render_plot,
            df=self.data_holder.param.filtered_df,
            x_col=self.x_select,
            y_col=self.y_select,
            color_col=self.color_select,
            size_col=self.size_select,
            palette=self.palette_select,
            alpha=self.alpha_slider,
        )

        return pn.Column(
            pn.pane.Markdown("### Scatter Plot"),
            controls,
            plot,
            sizing_mode="stretch_width",
        )
