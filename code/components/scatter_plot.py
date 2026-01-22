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
        self._sources: list[ColumnDataSource] = []
        self._source_ids: list[list[str]] = []
        self._pending_selection: list[str] | None = None
        self._url_sync_initialized = False
        self._syncing_selection = False
        self.data_holder.param.watch(self._sync_selection_to_source, "selected_record_ids")

    def _apply_source_selection(self, selected_ids: list[str], use_callback: bool = True) -> None:
        """Update scatter selection to match selected record IDs.

        Args:
            selected_ids: List of record IDs to select
            use_callback: If True, schedule via next_tick_callback (for async updates).
                          If False, set indices directly (for sync during figure creation).
        """
        if not self._sources:
            self._pending_selection = list(selected_ids)
            return
        selected_set = set(selected_ids)

        def set_indices():
            self._syncing_selection = True
            try:
                for source, id_values in zip(self._sources, self._source_ids):
                    indices = [
                        i for i, record_id in enumerate(id_values) if record_id in selected_set
                    ]
                    current = list(source.selected.indices or [])
                    if set(current) != set(indices):
                        source.selected.indices = indices
            finally:
                self._syncing_selection = False
                self._pending_selection = None

        if use_callback:
            doc = pn.state.curdoc
            if doc is not None:
                doc.add_next_tick_callback(set_indices)
            else:
                set_indices()
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
        location.sync(self.group_select, {"value": "sp_group"})
        location.sync(self.shape_select, {"value": "sp_shape"})
        location.sync(self.size_select, {"value": "sp_size"})
        location.sync(self.palette_select, {"value": "sp_palette"})
        location.sync(self.alpha_slider, {"value": "sp_alpha"})
        location.sync(self.width_slider, {"value": "sp_w"})
        location.sync(self.height_slider, {"value": "sp_h"})
        location.sync(self.font_size_slider, {"value": "sp_fs"})
        location.sync(self.size_range_slider, {"value": "sp_sr"})
        location.sync(self.size_gamma_slider, {"value": "sp_sg"})
        location.sync(self.size_uniform_slider, {"value": "sp_su"})

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
        self.group_select = pn.widgets.Select(
            name="Group By",
            options=["---"],
            value="---",
            width=180,
        )
        self.shape_select = pn.widgets.Select(
            name="Shape By",
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

        self.size_uniform_slider = pn.widgets.IntSlider(
            name="Point Size",
            start=scatter_config.min_size,
            end=scatter_config.max_size,
            step=1,
            value=scatter_config.default_size,
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

        self.size_select.param.watch(self._toggle_size_controls, "value")
        self._toggle_size_controls(None)

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

    def _get_xy_columns(self, df: pd.DataFrame) -> list[str]:
        """Get list of numeric or datetime columns for axes."""
        return df.select_dtypes(include=[np.number, "datetime"]).columns.tolist()

    def _get_all_columns(self, df: pd.DataFrame) -> list[str]:
        """Get list of all columns from DataFrame."""
        return df.columns.tolist()

    def _toggle_size_controls(self, _event) -> None:
        """Toggle size controls based on size-by selection."""
        is_uniform = self.size_select.value in ("---", None, "")
        self.size_uniform_slider.visible = is_uniform
        self.size_range_slider.visible = not is_uniform
        self.size_gamma_slider.visible = not is_uniform

    def _update_column_options(self, df: pd.DataFrame) -> None:
        """Update dropdown options based on available columns.

        Only sets default values when:
        - Options are being set for the first time (were empty)
        - Current value is no longer valid for new options
        """
        if df is None or df.empty:
            return

        numeric_cols = self._get_numeric_columns(df)
        xy_cols = self._get_xy_columns(df)
        all_cols = self._get_all_columns(df)
        scatter_config = self.config.scatter_plot
        group_cols = []
        for col in all_cols:
            series = df[col]
            if col in {"AIC", "BIC", "log_likelihood"}:
                print(f"Column {col} dtype: {series.dtype}")
            try:
                nunique = series.nunique(dropna=True)
            except TypeError:
                nunique = series.astype(str).nunique(dropna=True)
            is_categorical = series.dtype == "object" or series.dtype.name == "category"
            is_numeric = pd.api.types.is_numeric_dtype(series)
            if is_numeric and nunique < 20:
                group_cols.append(col)
            elif is_categorical:
                group_cols.append(col)

        # Track if this is initial setup (options were empty)
        x_was_empty = not self.x_select.options
        y_was_empty = not self.y_select.options
        color_was_empty = len(self.color_select.options) <= 1  # Only has "---"
        group_was_empty = len(self.group_select.options) <= 1
        shape_was_empty = len(self.shape_select.options) <= 1
        size_was_empty = len(self.size_select.options) <= 1

        # Update options
        self.x_select.options = xy_cols
        self.y_select.options = xy_cols
        self.color_select.options = ["---"] + all_cols
        self.group_select.options = ["---"] + group_cols
        self.shape_select.options = ["---"] + group_cols
        self.size_select.options = ["---"] + numeric_cols

        # Set X axis default only if needed
        if (x_was_empty and self.x_select.value in (None, "")) or (
            self.x_select.value not in xy_cols
        ):
            if scatter_config.x_column and scatter_config.x_column in xy_cols:
                self.x_select.value = scatter_config.x_column
            elif xy_cols:
                self.x_select.value = xy_cols[0]

        # Set Y axis default only if needed
        if (y_was_empty and self.y_select.value in (None, "")) or (
            self.y_select.value not in xy_cols
        ):
            if scatter_config.y_column and scatter_config.y_column in xy_cols:
                self.y_select.value = scatter_config.y_column
            elif len(xy_cols) > 1:
                self.y_select.value = xy_cols[1]
            elif xy_cols:
                self.y_select.value = xy_cols[0]

        # Set Color default only if needed
        color_options = ["---"] + all_cols
        if (color_was_empty and self.color_select.value in (None, "")) or (
            self.color_select.value not in color_options
        ):
            if scatter_config.color_column and scatter_config.color_column in all_cols:
                self.color_select.value = scatter_config.color_column
            else:
                self.color_select.value = "---"

        group_options = ["---"] + group_cols
        if (group_was_empty and self.group_select.value in (None, "")) or (
            self.group_select.value not in group_options
        ):
            self.group_select.value = "---"

        shape_options = ["---"] + group_cols
        if (shape_was_empty and self.shape_select.value in (None, "")) or (
            self.shape_select.value not in shape_options
        ):
            self.shape_select.value = "---"

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
            '<div style="max-width: 900px; padding: 10px; background: white; '
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
                'style="max-width: 700px; height: auto;" '
                "onerror=\"this.style.display='none'\">"
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
        group_col: str | None,
        shape_col: str | None,
        size_col: str | None,
        palette: str,
        alpha: float,
        plot_width: int,
        plot_height: int,
        font_size: int,
        size_range: tuple[int, int],
        size_gamma: float,
        uniform_size: int,
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
        group_column = group_col if group_col != "---" else None
        shape_column = shape_col if shape_col != "---" else None
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
            uniform_size,
        )

        x_is_datetime = pd.api.types.is_datetime64_any_dtype(df_valid[x_col])
        y_is_datetime = pd.api.types.is_datetime64_any_dtype(df_valid[y_col])
        x_values = (
            pd.to_datetime(df_valid[x_col], errors="coerce").to_numpy()
            if x_is_datetime
            else df_valid[x_col].values
        )
        y_values = (
            pd.to_datetime(df_valid[y_col], errors="coerce").to_numpy()
            if y_is_datetime
            else df_valid[y_col].values
        )

        # Prepare data for ColumnDataSource
        source_data = {
            "x": x_values,
            "y": y_values,
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
                df_valid[size_column].astype(str).values if size_column else ["N/A"] * len(df_valid)
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

        x_axis_type = "datetime" if x_is_datetime else "linear"
        y_axis_type = "datetime" if y_is_datetime else "linear"

        # Create figure
        p = figure(
            title=f"{y_col} vs {x_col}",
            x_axis_label=x_col,
            y_axis_label=y_col,
            width=int(plot_width),
            height=int(plot_height),
            x_axis_type=x_axis_type,
            y_axis_type=y_axis_type,
            tools="pan,wheel_zoom,box_zoom,box_select,lasso_select,reset,tap",
            active_drag="lasso_select",
            active_scroll="wheel_zoom",
        )

        markers = [
            "circle",
            "square",
            "triangle",
            "inverted_triangle",
            "diamond",
            "hex",
            "star",
            "plus",
            "circle_cross",
            "circle_x",
            "circle_dot",
            "square_cross",
            "square_x",
            "square_dot",
            "triangle_dot",
            "triangle_pin",
            "diamond_cross",
            "diamond_dot",
            "hex_dot",
            "star_dot",
            "asterisk",
            "cross",
            "dash",
            "dot",
            "square_pin",
            "x",
            "y",
        ]
        if shape_column and shape_column in df_valid.columns:
            shape_values = df_valid[shape_column].fillna("N/A").astype(str)
            unique_shapes = list(pd.unique(shape_values))
            shape_map = {
                value: markers[index % len(markers)] for index, value in enumerate(unique_shapes)
            }
            source_data[shape_column] = shape_values.values
            source_data["marker"] = shape_values.map(shape_map).values
        elif group_column and group_column in df_valid.columns:
            group_values = df_valid[group_column].fillna("N/A").astype(str)
            unique_groups = list(pd.unique(group_values))
            group_map = {
                value: markers[index % len(markers)] for index, value in enumerate(unique_groups)
            }
            source_data[group_column] = group_values.values
            source_data["marker"] = group_values.map(group_map).values
        else:
            source_data["marker"] = ["circle"] * len(df_valid)
        scatter_renderers = []

        shape_legend = []
        if (
            shape_column
            and shape_column in df_valid.columns
            and group_column
            and group_column in df_valid.columns
        ):
            shape_legend = [(shape, shape_map[shape]) for shape in unique_shapes]

        self._source = ColumnDataSource(data=source_data)
        self._sources = []
        self._source_ids = []

        if group_column and group_column in df_valid.columns:
            group_values = df_valid[group_column].fillna("N/A").astype(str)
            unique_groups = list(pd.unique(group_values))
            full_data = self._source.data
            for index, group in enumerate(unique_groups):
                mask = group_values == group
                if not mask.any():
                    continue
                group_data = {key: np.asarray(values)[mask] for key, values in full_data.items()}
                group_source = ColumnDataSource(data=group_data)
                renderer = p.scatter(
                    x="x",
                    y="y",
                    source=group_source,
                    size="size",
                    alpha=alpha,
                    color=color_spec,
                    marker="marker",
                    legend_label=str(group),
                    line_color="#333333",
                    line_width=0.5,
                    selection_line_color="black",
                    selection_line_width=2,
                )
                scatter_renderers.append(renderer)
                self._sources.append(group_source)
                self._source_ids.append(
                    [str(value) for value in group_source.data.get(self.config.id_column, [])]
                )
        else:
            scatter_kwargs = {}
            if shape_column and shape_column in df_valid.columns:
                scatter_kwargs["legend_field"] = shape_column

            scatter_renderer = p.scatter(
                x="x",
                y="y",
                source=self._source,
                size="size",
                alpha=alpha,
                color=color_spec,
                marker="marker",
                line_color="#333333",
                line_width=0.5,
                selection_line_color="black",
                selection_line_width=2,
                **scatter_kwargs,
            )
            scatter_renderers.append(scatter_renderer)
            self._sources.append(self._source)
            self._source_ids.append(
                [str(value) for value in self._source.data.get(self.config.id_column, [])]
            )

        if shape_legend:
            for shape_label, shape_marker in shape_legend:
                legend_renderer = p.scatter(
                    x=[np.nan],
                    y=[np.nan],
                    marker=shape_marker,
                    size=14,
                    color="#666666",
                    alpha=0.8,
                    legend_label=str(shape_label),
                )
                scatter_renderers.append(legend_renderer)

        # Add hover tool
        hover = HoverTool(
            tooltips=self._build_tooltip_html(),
            renderers=scatter_renderers,
            attachment="right",
        )
        p.add_tools(hover)

        # Selection callback (handles lasso, box select, and tap)
        def on_tap_select(_attr, _old, _new):
            if self._syncing_selection:
                return
            selected_ids = []
            for source in self._sources:
                id_values = source.data.get(self.config.id_column, [])
                for idx in source.selected.indices or []:
                    if idx < len(id_values):
                        selected_ids.append(str(id_values[idx]))
            logger.debug(f"Selected records: {selected_ids}")
            if selected_ids != self.data_holder.selected_record_ids:
                self.data_holder.selected_record_ids = selected_ids

        for source in self._sources:
            source.selected.on_change("indices", on_tap_select)

        # Apply selection synchronously during figure creation (use_callback=False)
        # This ensures selection is set before the figure is returned
        selected_ids = [str(record_id) for record_id in self.data_holder.selected_record_ids]
        if selected_ids:
            self._apply_source_selection(selected_ids, use_callback=False)
        elif self._pending_selection:
            self._apply_source_selection(self._pending_selection, use_callback=False)

        def reapply_selection() -> None:
            ids = [str(record_id) for record_id in self.data_holder.selected_record_ids]
            if ids:
                self._apply_source_selection(ids, use_callback=False)
            elif self._pending_selection:
                self._apply_source_selection(self._pending_selection, use_callback=False)

        doc = pn.state.curdoc
        if doc is not None:
            doc.add_next_tick_callback(reapply_selection)

        # Add color bar for continuous mappings
        if color_column:
            add_color_bar(p, color_mapper, color_column, font_size=font_size)

        if scatter_renderers and p.legend:
            legend = p.legend[0]
            legend.click_policy = "hide"
            legend.location = "center"
            p.add_layout(legend, "right")

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
        group_col: str,
        shape_col: str,
        size_col: str,
        palette: str,
        alpha: float,
        plot_width: int,
        plot_height: int,
        font_size: int,
        size_range: tuple[int, int],
        size_gamma: float,
        uniform_size: int,
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
            group_col = self.group_select.value or group_col
            shape_col = self.shape_select.value or shape_col
            size_col = self.size_select.value or size_col

            return self._create_figure(
                df,
                x_col,
                y_col,
                color_col,
                group_col,
                shape_col,
                size_col,
                palette,
                alpha,
                plot_width,
                plot_height,
                font_size,
                size_range,
                size_gamma,
                uniform_size,
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
            self.size_uniform_slider,
            self.size_range_slider,
            self.size_gamma_slider,
            pn.layout.Divider(),
            # Shape settings
            self.shape_select,
            pn.layout.Divider(),
            self.group_select,
            pn.layout.Divider(),
            # Plot settings
            pn.Card(
                self.alpha_slider,
                self.width_slider,
                self.height_slider,
                self.font_size_slider,
                title="More settings",
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
            group_col=self.group_select,
            shape_col=self.shape_select,
            size_col=self.size_select,
            palette=self.palette_select,
            alpha=self.alpha_slider,
            plot_width=self.width_slider,
            plot_height=self.height_slider,
            font_size=self.font_size_slider,
            size_range=self.size_range_slider,
            size_gamma=self.size_gamma_slider,
            uniform_size=self.size_uniform_slider,
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
