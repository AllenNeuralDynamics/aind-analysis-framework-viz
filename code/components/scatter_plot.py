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
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    LinearColorMapper,
    TapTool,
)
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
        self._url_sync_initialized = False
        self._syncing_selection = False
        self.data_holder.param.watch(self._sync_selection_to_source, "selected_record_ids")

    def _apply_source_selection(self, selected_ids: list[str]) -> None:
        """Update scatter selection to match selected record IDs."""
        if self._source is None:
            return
        id_values = [str(value) for value in self._source.data.get(self.config.id_column, [])]
        indices = [i for i, record_id in enumerate(id_values) if record_id in selected_ids]
        current = list(self._source.selected.indices or [])
        if set(current) == set(indices):
            return
        def set_indices():
            self._syncing_selection = True
            try:
                self._source.selected.indices = indices
            finally:
                self._syncing_selection = False

        doc = pn.state.curdoc
        if doc is not None:
            doc.add_next_tick_callback(set_indices)
        else:
            set_indices()

    def _sync_selection_to_source(self, event) -> None:
        """Sync DataHolder selection into the scatter plot."""
        selected_ids = [str(record_id) for record_id in (event.new or [])]
        self._apply_source_selection(selected_ids)

    def _sync_url_state(self) -> None:
        """Bidirectionally sync widget state to URL query params."""
        if self._url_sync_initialized:
            return
        self._url_sync_initialized = True

        location = pn.state.location
        location.sync(self.x_select, {"value": "sp_x"})
        location.sync(self.y_select, {"value": "sp_y"})
        location.sync(self.color_select, {"value": "sp_color"})
        location.sync(self.size_select, {"value": "sp_size"})
        location.sync(self.palette_select, {"value": "sp_palette"})
        location.sync(self.alpha_slider, {"value": "sp_alpha"})
        location.sync(self.width_slider, {"value": "sp_w"})
        location.sync(self.height_slider, {"value": "sp_h"})
        location.sync(self.font_size_slider, {"value": "sp_fs"})
        location.sync(self.size_range_slider, {"value": "sp_sr"})
        location.sync(self.size_gamma_slider, {"value": "sp_sg"})

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
            options=["---"],
            value="---",
            width=180,
        )
        self.size_select = pn.widgets.Select(
            name="Size By",
            options=["---"],
            value="---",
            width=180,
        )

        self.size_range_slider = pn.widgets.RangeSlider(
            name="Size Range",
            start=0,
            end=40,
            step=1,
            value=(scatter_config.min_size, scatter_config.max_size),
            width=180,
        )
        self.size_gamma_slider = pn.widgets.FloatSlider(
            name="Size Gamma",
            start=0.0,
            end=3.0,
            step=0.1,
            value=scatter_config.default_gamma,
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

        # Plot settings
        self.width_slider = pn.widgets.IntSlider(
            name="Width",
            start=300,
            end=1600,
            step=50,
            value=scatter_config.width,
            width=180,
        )
        self.height_slider = pn.widgets.IntSlider(
            name="Height",
            start=250,
            end=1200,
            step=50,
            value=scatter_config.height,
            width=180,
        )
        self.font_size_slider = pn.widgets.IntSlider(
            name="Font Size",
            start=8,
            end=24,
            step=1,
            value=scatter_config.font_size,
            width=180,
        )

    def _get_numeric_columns(self, df: pd.DataFrame) -> list[str]:
        """Get list of numeric columns from DataFrame."""
        return df.select_dtypes(include=[np.number]).columns.tolist()

    def _get_all_columns(self, df: pd.DataFrame) -> list[str]:
        """Get list of all columns from DataFrame."""
        return df.columns.tolist()

    def _update_column_options(self, df: pd.DataFrame) -> None:
        """Update dropdown options based on available columns.

        Only sets default values when:
        - Options are being set for the first time (were empty)
        - Current value is no longer valid for new options
        """
        if df is None or df.empty:
            return

        numeric_cols = self._get_numeric_columns(df)
        all_cols = self._get_all_columns(df)
        scatter_config = self.config.scatter_plot

        # Track if this is initial setup (options were empty)
        x_was_empty = not self.x_select.options
        y_was_empty = not self.y_select.options
        color_was_empty = len(self.color_select.options) <= 1  # Only has "---"
        size_was_empty = len(self.size_select.options) <= 1

        # Update options
        self.x_select.options = numeric_cols
        self.y_select.options = numeric_cols
        self.color_select.options = ["---"] + all_cols
        self.size_select.options = ["---"] + numeric_cols

        # Set X axis default only if needed
        if (x_was_empty and self.x_select.value in (None, "")) or (
            self.x_select.value not in numeric_cols
        ):
            if scatter_config.x_column and scatter_config.x_column in numeric_cols:
                self.x_select.value = scatter_config.x_column
            elif numeric_cols:
                self.x_select.value = numeric_cols[0]

        # Set Y axis default only if needed
        if (y_was_empty and self.y_select.value in (None, "")) or (
            self.y_select.value not in numeric_cols
        ):
            if scatter_config.y_column and scatter_config.y_column in numeric_cols:
                self.y_select.value = scatter_config.y_column
            elif len(numeric_cols) > 1:
                self.y_select.value = numeric_cols[1]
            elif numeric_cols:
                self.y_select.value = numeric_cols[0]

        # Set Color default only if needed
        color_options = ["---"] + all_cols
        if (color_was_empty and self.color_select.value in (None, "")) or (
            self.color_select.value not in color_options
        ):
            if scatter_config.color_column and scatter_config.color_column in all_cols:
                self.color_select.value = scatter_config.color_column
            else:
                self.color_select.value = "---"

        # Set Size default only if needed
        size_options = ["---"] + numeric_cols
        if (size_was_empty and self.size_select.value in (None, "")) or (
            self.size_select.value not in size_options
        ):
            if scatter_config.size_column and scatter_config.size_column in numeric_cols:
                self.size_select.value = scatter_config.size_column
            else:
                self.size_select.value = "---"

    def _build_tooltip_html(self) -> str:
        """Build HTML template for hover tooltip."""
        scatter_config = self.config.scatter_plot
        id_col = self.config.id_column

        # Basic info tooltip
        tooltip_parts = [
            '<div style="max-width: 600px; padding: 10px; background: white; '
            'border: 1px solid #ccc; border-radius: 4px;">',
            f'<div style="font-weight: bold; margin-bottom: 5px;">@{{{id_col}}}</div>',
            '<div style="font-size: 12px; color: #666;">',
            "X: @x{0.000} | Y: @y{0.000}",
            "</div>",
        ]

        # Add color/size info if set
        tooltip_parts.append(
            '<div style="font-size: 12px; color: #666; margin-top: 3px;">'
            "Color: @color_col | Size: @size_col"
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
        plot_width: int,
        plot_height: int,
        font_size: int,
        size_range: tuple[int, int],
        size_gamma: float,
    ):
        """Create the Bokeh scatter plot figure."""
        scatter_config = self.config.scatter_plot

        if df is None or df.empty or not x_col or not y_col:
            return pn.pane.Markdown(
                "No data available or axes not selected.",
                css_classes=["alert", "alert-info", "p-3"],
            )

        # Check that columns exist in DataFrame
        if x_col not in df.columns or y_col not in df.columns:
            return pn.pane.Markdown(
                f"Selected columns not found in data. Please select valid X and Y axes.",
                css_classes=["alert", "alert-warning", "p-3"],
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
        color_column = color_col if color_col != "---" else None
        color_spec, color_mapper = determine_color_mapping(
            df_valid,
            color_column,
            palette,
            scatter_config.max_categorical_values,
        )  # Now uses intelligent palette selection from all_palettes

        # Determine size mapping
        size_column = size_col if size_col != "---" else None
        min_size, max_size = size_range
        min_size = int(min_size)
        max_size = int(max_size)
        if min_size > max_size:
            min_size, max_size = max_size, min_size

        sizes = determine_size_mapping(
            df_valid,
            size_column,
            min_size,
            max_size,
            size_gamma,
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
            if isinstance(color_mapper, LinearColorMapper):
                source_data[color_column] = pd.to_numeric(
                    df_valid[color_column], errors="coerce"
                ).values
            else:
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
            width=int(plot_width),
            height=int(plot_height),
            tools="pan,wheel_zoom,box_zoom,box_select,lasso_select,reset,tap",
            active_drag="lasso_select",
            active_scroll="wheel_zoom",
        )

        # Add scatter points
        scatter_renderer = p.scatter(
            x="x",
            y="y",
            source=self._source,
            size="size",
            alpha=alpha,
            color=color_spec,
            line_color=None,
            selection_alpha=1.0,
            selection_line_color="black",
            selection_line_width=2,
        )

        # Add hover tool
        hover = HoverTool(
            tooltips=self._build_tooltip_html(),
            renderers=[scatter_renderer],
            attachment="right",
        )
        p.add_tools(hover)

        # Selection callback (handles lasso, box select, and tap)
        def on_tap_select(_attr, _old, new):
            if self._syncing_selection:
                return
            if new:
                selected_ids = []
                id_values = self._source.data[self.config.id_column]
                for idx in new:
                    if idx < len(id_values):
                        selected_ids.append(str(id_values[idx]))
                logger.debug(f"Selected records: {selected_ids}")
                if selected_ids != self.data_holder.selected_record_ids:
                    self.data_holder.selected_record_ids = selected_ids
            else:
                if self.data_holder.selected_record_ids:
                    self.data_holder.selected_record_ids = []

        self._source.selected.on_change("indices", on_tap_select)
        self._apply_source_selection(
            [str(record_id) for record_id in self.data_holder.selected_record_ids]
        )

        # Add color bar for continuous mappings
        if color_column:
            add_color_bar(p, color_mapper, color_column, font_size=font_size)

        # Style the plot
        title_size = max(8, int(font_size) + 2)
        label_size = max(8, int(font_size))
        tick_size = max(6, int(font_size) - 2)
        p.title.text_font_size = f"{title_size}pt"
        p.xaxis.axis_label_text_font_size = f"{label_size}pt"
        p.yaxis.axis_label_text_font_size = f"{label_size}pt"
        p.xaxis.major_label_text_font_size = f"{tick_size}pt"
        p.yaxis.major_label_text_font_size = f"{tick_size}pt"

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
        plot_width: int,
        plot_height: int,
        font_size: int,
        size_range: tuple[int, int],
        size_gamma: float,
    ):
        """Render the scatter plot with current settings."""
        try:
            # Update column options when data changes
            self._update_column_options(df)

            # After updating options, use current widget values to handle
            # the race condition where widget values are set during render
            x_col = self.x_select.value or x_col
            y_col = self.y_select.value or y_col
            color_col = self.color_select.value or color_col
            size_col = self.size_select.value or size_col

            return self._create_figure(
                df,
                x_col,
                y_col,
                color_col,
                size_col,
                palette,
                alpha,
                plot_width,
                plot_height,
                font_size,
                size_range,
                size_gamma,
            )
        except Exception as e:
            logger.error(f"Error rendering scatter plot: {e}")
            return pn.pane.Markdown(
                f"Error rendering scatter plot: {e}",
                css_classes=["alert", "alert-danger", "p-3"],
            )

    def create(self) -> pn.viewable.Viewable:
        """
        Create the scatter plot component with controls.

        Returns:
            Panel viewable with controls and reactive plot
        """
        # Control sidebar - grouped logically
        controls = pn.Column(
            # Axis selection
            self.x_select,
            self.y_select,
            pn.layout.Divider(),
            # Color settings
            self.color_select,
            self.palette_select,
            pn.layout.Divider(),
            # Size settings
            self.size_select,
            pn.Card(
                self.size_range_slider,
                self.size_gamma_slider,
                title="Size settings",
                collapsed=True,
                sizing_mode="stretch_width",
            ),
            pn.layout.Divider(),
            # Plot settings
            pn.Card(
                self.alpha_slider,
                self.width_slider,
                self.height_slider,
                self.font_size_slider,
                title="Plot settings",
                collapsed=True,
                sizing_mode="stretch_width",
            ),
            width=200,
        )

        self._sync_url_state()

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
            plot_width=self.width_slider,
            plot_height=self.height_slider,
            font_size=self.font_size_slider,
            size_range=self.size_range_slider,
            size_gamma=self.size_gamma_slider,
        )

        # Side-by-side layout: controls on left, plot on right
        return pn.Row(
            controls,
            pn.Spacer(width=20),
            pn.Column(
                pn.pane.Markdown("### Scatter Plot"),
                plot,
                sizing_mode="stretch_width",
            ),
            sizing_mode="stretch_width",
        )
