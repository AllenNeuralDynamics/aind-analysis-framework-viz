"""
Pair plot component for exploring pairwise correlations across multiple columns.

Renders an N×N grid of scatter plots (off-diagonal) and histograms (diagonal)
using Bokeh gridplot. Reuses color/size mapping and aggregation logic from
the scatter plot component.
"""

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import panel as pn
from bokeh.layouts import gridplot
from bokeh.models import (
    Band,
    CategoricalColorMapper,
    ColumnDataSource,
    HoverTool,
    LinearColorMapper,
    Range1d,
)
from bokeh.plotting import figure

from .base import BaseComponent
from .color_mapping import add_color_bar, determine_color_mapping
from .scatter_plot import compute_aggregation
from .size_mapping import determine_size_mapping

if TYPE_CHECKING:
    from config import AppConfig
    from core.base_app import DataHolder

logger = logging.getLogger(__name__)


class PairPlot(BaseComponent):
    """
    Pair plot component showing pairwise scatter plots and diagonal histograms.

    Features:
    - Select 2-5 numeric columns for pairwise comparison
    - Shared color/size mapping across all cells
    - Diagonal histograms with optional KDE
    - Aggregation overlays (linear fit, lowess, etc.)
    - Selection syncing across cells and to DataHolder
    """

    def __init__(self, data_holder: "DataHolder", config: "AppConfig"):
        super().__init__(data_holder, config)
        self._init_controls()
        self._sources: list[ColumnDataSource] = []
        self._source_ids: list[list[str]] = []
        self._pending_selection: list[str] | None = None
        self._url_sync_initialized = False
        self._syncing_selection = False
        self.data_holder.param.watch(self._sync_selection_to_source, "selected_record_ids")

    def _apply_source_selection(self, selected_ids: list[str], use_callback: bool = True) -> None:
        """Update all scatter source selections to match selected record IDs."""
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
        """Sync DataHolder selection into pair plot sources."""
        selected_ids = [str(record_id) for record_id in (event.new or [])]
        self._apply_source_selection(selected_ids)

    def _sync_url_state(self) -> None:
        """Bidirectionally sync widget state to URL query params."""
        if self._url_sync_initialized:
            return
        self._url_sync_initialized = True

        location = pn.state.location
        location.sync(self.columns_select, {"value": "pp_cols"})
        location.sync(self.color_select, {"value": "pp_color"})
        location.sync(self.palette_select, {"value": "pp_palette"})
        location.sync(self.reverse_colors_toggle, {"value": "pp_reverse"})
        location.sync(self.group_select, {"value": "pp_group"})
        location.sync(self.group_applies_to, {"value": "pp_gapply"})
        location.sync(self.size_select, {"value": "pp_size"})
        location.sync(self.size_uniform_slider, {"value": "pp_su"})
        location.sync(self.size_range_slider, {"value": "pp_sr"})
        location.sync(self.size_gamma_slider, {"value": "pp_sg"})
        location.sync(self.alpha_slider, {"value": "pp_alpha"})
        location.sync(self.aggr_toggle, {"value": "pp_aggr"})
        location.sync(self.aggr_method, {"value": "pp_aggrm"})
        location.sync(self.aggr_quantiles, {"value": "pp_aggrq"})
        location.sync(self.aggr_n_quantiles, {"value": "pp_aggrnq"})
        location.sync(self.aggr_smooth, {"value": "pp_aggrs"})
        location.sync(self.aggr_line_width_slider, {"value": "pp_alw"})
        location.sync(self.hist_bins_slider, {"value": "pp_hbin"})
        location.sync(self.hist_kde_toggle, {"value": "pp_kde"})
        location.sync(self.width_slider, {"value": "pp_w"})
        location.sync(self.font_size_slider, {"value": "pp_fs"})
        location.sync(self.hide_dots, {"value": "pp_hdots"})
        location.sync(self.hide_color_bar, {"value": "pp_hcb"})
        location.sync(self.hide_legend, {"value": "pp_hleg"})

    def _init_controls(self) -> None:
        """Initialize control widgets."""
        scatter_config = self.config.scatter_plot

        self.columns_select = pn.widgets.MultiSelect(
            name="Columns (2-5)",
            options=[],
            value=[],
            size=8,
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
        self.group_applies_to = pn.widgets.Select(
            name="Group applies to",
            options=["Color"],
            value="Color",
            width=180,
            visible=False,
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

        self.palette_select = pn.widgets.Select(
            name="Color Palette",
            options=scatter_config.color_palettes,
            value=scatter_config.color_palettes[0],
            width=180,
        )
        self.reverse_colors_toggle = pn.widgets.Toggle(
            name="Reverse colors",
            value=False,
            button_type="light",
            width=180,
        )
        self.color_select.param.watch(self._toggle_color_controls, "value")
        self._toggle_color_controls(None)

        self.group_select.param.watch(self._toggle_group_applies, "value")
        self.group_applies_to.param.watch(self._toggle_group_applies, "value")
        self._toggle_group_applies(None)

        self.alpha_slider = pn.widgets.FloatSlider(
            name="Opacity",
            start=0.05,
            end=1.0,
            step=0.05,
            value=scatter_config.default_alpha,
            width=180,
        )

        # Aggregation controls
        aggr_methods = ["---", "mean", "mean +/- sem", "lowess", "running average", "linear fit"]
        self.aggr_toggle = pn.widgets.Checkbox(
            name="Aggregation",
            value=False,
            width=180,
        )
        self.aggr_method = pn.widgets.Select(
            name="Method",
            options=aggr_methods,
            value="linear fit",
            width=180,
        )
        self.aggr_quantiles = pn.widgets.Checkbox(
            name="Use quantiles of x",
            value=False,
            width=180,
        )
        self.aggr_n_quantiles = pn.widgets.IntSlider(
            name="Number of bins",
            start=1,
            end=100,
            step=1,
            value=20,
            width=180,
        )
        self.aggr_smooth = pn.widgets.IntSlider(
            name="Smooth factor",
            start=1,
            end=20,
            step=1,
            value=5,
            width=180,
        )
        self.aggr_line_width_slider = pn.widgets.FloatSlider(
            name="Aggr line width",
            start=0.5,
            end=10.0,
            step=0.5,
            value=2.0,
            width=180,
        )
        for w in [self.aggr_toggle, self.aggr_method, self.aggr_quantiles,
                   self.aggr_n_quantiles, self.aggr_smooth]:
            w.param.watch(self._toggle_aggr_controls, "value")
        self._toggle_aggr_controls(None)

        # Histogram controls
        self.hist_bins_slider = pn.widgets.IntSlider(
            name="Histogram bins",
            start=5,
            end=100,
            step=1,
            value=25,
            width=180,
        )
        self.hist_kde_toggle = pn.widgets.Checkbox(
            name="KDE curve",
            value=False,
            width=180,
        )

        # Plot settings
        self.width_slider = pn.widgets.IntSlider(
            name="Total width",
            start=400,
            end=1600,
            step=50,
            value=900,
            width=180,
        )
        self.font_size_slider = pn.widgets.IntSlider(
            name="Font Size",
            start=6,
            end=20,
            step=1,
            value=10,
            width=180,
        )
        self.hide_dots = pn.widgets.Checkbox(
            name="Hide dots",
            value=False,
            width=180,
        )
        self.hide_color_bar = pn.widgets.Checkbox(
            name="Hide color bar",
            value=False,
            width=180,
        )
        self.hide_legend = pn.widgets.Checkbox(
            name="Hide legend",
            value=False,
            width=180,
        )
        self.clear_selection_button = pn.widgets.Button(
            name="Clear Selection",
            button_type="warning",
            width=180,
        )
        self.clear_selection_button.on_click(self._clear_selection)

    def _toggle_size_controls(self, _event) -> None:
        is_uniform = self.size_select.value in ("---", None, "")
        self.size_uniform_slider.visible = is_uniform
        self.size_range_slider.visible = not is_uniform
        self.size_gamma_slider.visible = not is_uniform

    def _toggle_color_controls(self, _event) -> None:
        has_group = self.group_select.value not in ("---", None, "")
        applies = self.group_applies_to.value
        group_overrides_color = has_group and applies in ("Color",)
        has_color = self.color_select.value not in ("---", None, "")
        show_controls = has_color or group_overrides_color
        self.palette_select.visible = show_controls
        self.reverse_colors_toggle.visible = show_controls

    def _toggle_group_applies(self, _event) -> None:
        has_group = self.group_select.value not in ("---", None, "")
        self.group_applies_to.visible = has_group
        overrides_color = has_group and self.group_applies_to.value == "Color"
        self.color_select.disabled = overrides_color
        self._toggle_color_controls(None)

    def _toggle_aggr_controls(self, _event) -> None:
        smooth_methods = {"lowess", "running average"}
        binned_methods = {"mean", "mean +/- sem"}
        on = self.aggr_toggle.value
        method = self.aggr_method.value
        self.aggr_method.visible = on
        self.aggr_smooth.visible = on and method in smooth_methods
        needs_bins = on and method in binned_methods
        self.aggr_quantiles.visible = needs_bins
        self.aggr_n_quantiles.visible = needs_bins
        self.aggr_line_width_slider.visible = on
        if self.aggr_quantiles.value:
            self.aggr_n_quantiles.name = "Number of quantiles"
        else:
            self.aggr_n_quantiles.name = "Number of bins"

    def _clear_selection(self, _event) -> None:
        """Clear all selections across pair plot and DataHolder."""
        self._syncing_selection = True
        try:
            for source in self._sources:
                source.selected.indices = []
        finally:
            self._syncing_selection = False
        self.data_holder.selected_record_ids = []

    def _update_column_options(self, df: pd.DataFrame) -> None:
        """Update dropdown options based on available columns."""
        if df is None or df.empty:
            return

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        all_cols = df.columns.tolist()
        group_cols = []
        for col in all_cols:
            series = df[col]
            try:
                nunique = series.nunique(dropna=True)
            except TypeError:
                nunique = series.astype(str).nunique(dropna=True)
            if nunique < 30:
                group_cols.append(col)

        cols_was_empty = not self.columns_select.options
        color_was_empty = len(self.color_select.options) <= 1
        group_was_empty = len(self.group_select.options) <= 1
        size_was_empty = len(self.size_select.options) <= 1

        self.columns_select.options = numeric_cols
        self.color_select.options = ["---"] + all_cols
        self.group_select.options = ["---"] + group_cols
        self.size_select.options = ["---"] + numeric_cols

        # Set default column selection
        if cols_was_empty and not self.columns_select.value:
            self.columns_select.value = numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols[:2]

        # Validate current values
        if self.columns_select.value:
            valid = [c for c in self.columns_select.value if c in numeric_cols]
            if valid != self.columns_select.value:
                self.columns_select.value = valid if valid else numeric_cols[:3]

        color_options = ["---"] + all_cols
        if (color_was_empty and self.color_select.value in (None, "")) or (
            self.color_select.value not in color_options
        ):
            self.color_select.value = "---"

        group_options = ["---"] + group_cols
        if (group_was_empty and self.group_select.value in (None, "")) or (
            self.group_select.value not in group_options
        ):
            self.group_select.value = "---"

        size_options = ["---"] + numeric_cols
        if (size_was_empty and self.size_select.value in (None, "")) or (
            self.size_select.value not in size_options
        ):
            self.size_select.value = "---"

    def _create_pair_grid(
        self,
        df: pd.DataFrame,
        columns: list[str],
        color_col: str | None,
        group_col: str | None,
        size_col: str | None,
        palette: str,
        reverse_colors: bool,
        alpha: float,
        total_width: int,
        font_size: int,
        size_range: tuple[int, int],
        size_gamma: float,
        uniform_size: int,
        aggr_on: bool,
        aggr_method: str,
        aggr_quantiles: bool,
        aggr_n_quantiles: int,
        aggr_smooth: int,
        aggr_line_width: float,
        hist_bins: int,
        hist_kde: bool,
        hide_dots: bool,
        hide_color_bar: bool,
        hide_legend: bool,
    ):
        """Create the N×N pair plot grid."""
        scatter_config = self.config.scatter_plot

        if df is None or df.empty or len(columns) < 2:
            return pn.pane.Markdown(
                "Select at least 2 numeric columns to create a pair plot.",
                css_classes=["alert", "alert-info", "p-3"],
            )

        # Limit to 5 columns
        columns = columns[:5]
        n = len(columns)

        # Filter to rows with valid values for all selected columns
        df_valid = df[columns].dropna()
        if df_valid.empty:
            return pn.pane.Markdown(
                "No valid data for selected columns.",
                css_classes=["alert", "alert-warning", "p-3"],
            )
        df_valid = df.loc[df_valid.index].copy()

        # Determine color mapping
        color_column = color_col if color_col != "---" else None
        group_column = group_col if group_col != "---" else None

        # Apply group override
        if group_column:
            color_column = group_column

        color_spec, color_mapper = determine_color_mapping(
            df_valid,
            color_column,
            palette,
            scatter_config.max_categorical_values,
            reverse_colors,
        )

        # Determine size mapping
        size_column = size_col if size_col != "---" else None
        min_size, max_size = size_range
        min_size, max_size = int(min_size), int(max_size)
        if min_size > max_size:
            min_size, max_size = max_size, min_size
        sizes = determine_size_mapping(df_valid, size_column, min_size, max_size, size_gamma, uniform_size)

        # Build group info for coloring histograms
        is_categorical = (
            color_column
            and color_column in df_valid.columns
            and isinstance(color_mapper, CategoricalColorMapper)
        )
        if is_categorical:
            factors = color_mapper.factors
            pal = color_mapper.palette
            color_map = dict(zip(factors, pal))
            cat_values = df_valid[color_column].fillna("N/A").astype(str)
            groups = [(cat, color_map.get(cat, "#808080")) for cat in factors]
        else:
            groups = [("all", "#3B82F6")]
            cat_values = pd.Series(["all"] * len(df_valid), index=df_valid.index)

        cell_size = max(150, total_width // n)
        tick_size = max(6, font_size - 2)

        # Create shared ranges for each column (based on diagonal)
        ranges = {}
        for col in columns:
            vals = pd.to_numeric(df_valid[col], errors="coerce").dropna()
            if len(vals) > 0:
                pad = (vals.max() - vals.min()) * 0.05 if vals.max() != vals.min() else 1.0
                ranges[col] = Range1d(start=vals.min() - pad, end=vals.max() + pad)
            else:
                ranges[col] = Range1d(start=0, end=1)

        self._sources = []
        self._source_ids = []

        grid_rows = []
        legend_added = False

        for row_idx in range(n):
            row_figures = []
            for col_idx in range(n):
                x_col = columns[col_idx]
                y_col = columns[row_idx]

                # Determine axis visibility
                show_x_label = row_idx == n - 1
                show_y_label = col_idx == 0

                if row_idx == col_idx:
                    # Diagonal: histogram
                    p = self._create_histogram_cell(
                        df_valid, x_col, cat_values, groups,
                        ranges[x_col], cell_size, hist_bins, hist_kde,
                        show_x_label, show_y_label, font_size, tick_size,
                    )
                else:
                    # Off-diagonal: scatter plot
                    p = self._create_scatter_cell(
                        df_valid, x_col, y_col, color_spec, color_column,
                        color_mapper, sizes, alpha, ranges[x_col], ranges[y_col],
                        cell_size, font_size, tick_size, show_x_label, show_y_label,
                        hide_dots, group_column, groups,
                        aggr_on, aggr_method, aggr_quantiles, aggr_n_quantiles,
                        aggr_smooth, aggr_line_width,
                    )

                row_figures.append(p)
            grid_rows.append(row_figures)

        # Add color bar to top-right cell if applicable
        if color_column and not hide_color_bar and color_mapper:
            # Add to top-right corner cell
            add_color_bar(grid_rows[0][-1], color_mapper, color_column, font_size=font_size)

        layout = gridplot(grid_rows, merge_tools=True)

        # Apply pending selections
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

        return pn.pane.Bokeh(layout, sizing_mode="stretch_width")

    def _create_histogram_cell(
        self, df_valid, col, cat_values, groups,
        x_range, cell_size, n_bins, show_kde,
        show_x_label, show_y_label, font_size, tick_size,
    ):
        """Create a histogram for a diagonal cell."""
        p = figure(
            frame_width=cell_size,
            frame_height=cell_size,
            x_range=x_range,
            tools="wheel_zoom",
            toolbar_location=None,
            active_scroll="wheel_zoom",
            title=col,
        )

        vals = pd.to_numeric(df_valid[col], errors="coerce")
        valid_mask = vals.notna()
        if valid_mask.sum() < 2:
            return p

        all_vals = vals[valid_mask].values
        edges = np.histogram_bin_edges(all_vals, bins=n_bins)
        hist_alpha = 0.6
        bottom = np.zeros(len(edges) - 1)

        for cat, color in groups:
            mask = (cat_values == cat) & valid_mask
            if not mask.any():
                continue
            hist, _ = np.histogram(vals[mask].values, bins=edges)
            hist = hist.astype(float)
            p.quad(
                top=bottom + hist,
                bottom=bottom,
                left=edges[:-1],
                right=edges[1:],
                fill_color=color,
                line_color="white",
                fill_alpha=hist_alpha,
            )
            bottom = bottom + hist

        if show_kde:
            for cat, color in groups:
                mask = (cat_values == cat) & valid_mask
                cat_vals = vals[mask].values
                if len(cat_vals) < 2:
                    continue
                try:
                    from scipy.stats import gaussian_kde
                    kde = gaussian_kde(cat_vals)
                    x_grid = np.linspace(all_vals.min(), all_vals.max(), 200)
                    density = kde(x_grid)
                    bin_width = edges[1] - edges[0]
                    density = density * len(cat_vals) * bin_width
                    p.line(x_grid, density, line_color=color, line_width=2)
                except Exception:
                    pass

        # Style
        p.title.text_font_size = f"{font_size}pt"
        if not show_x_label:
            p.xaxis.axis_label = ""
            p.xaxis.major_label_text_font_size = "0pt"
        else:
            p.xaxis.axis_label = col
            p.xaxis.axis_label_text_font_size = f"{font_size}pt"
            p.xaxis.major_label_text_font_size = f"{tick_size}pt"
        if not show_y_label:
            p.yaxis.axis_label = ""
            p.yaxis.major_label_text_font_size = "0pt"
        else:
            p.yaxis.axis_label = "Count"
            p.yaxis.axis_label_text_font_size = f"{font_size}pt"
            p.yaxis.major_label_text_font_size = f"{tick_size}pt"

        p.min_border = 5
        return p

    def _create_scatter_cell(
        self, df_valid, x_col, y_col, color_spec, color_column,
        color_mapper, sizes, alpha, x_range, y_range,
        cell_size, font_size, tick_size, show_x_label, show_y_label,
        hide_dots, group_column, groups,
        aggr_on, aggr_method, aggr_quantiles, aggr_n_quantiles,
        aggr_smooth, aggr_line_width,
    ):
        """Create a scatter plot for an off-diagonal cell."""
        p = figure(
            frame_width=cell_size,
            frame_height=cell_size,
            x_range=x_range,
            y_range=y_range,
            tools="pan,wheel_zoom,box_select,lasso_select,reset,tap",
            active_drag="lasso_select",
            active_scroll="wheel_zoom",
            toolbar_location=None,
        )

        id_col = self.config.id_column

        # Build source data
        source_data = {
            "x": df_valid[x_col].values,
            "y": df_valid[y_col].values,
            "size": sizes,
            id_col: df_valid[id_col].astype(str).values,
        }

        # Add color column data
        if color_column and color_column in df_valid.columns:
            if isinstance(color_mapper, LinearColorMapper):
                source_data[color_column] = pd.to_numeric(
                    df_valid[color_column], errors="coerce"
                ).values
            else:
                source_data[color_column] = df_valid[color_column].astype(str).values

        scatter_renderers = []

        if group_column and group_column in df_valid.columns:
            group_values = df_valid[group_column].fillna("N/A").astype(str)
            unique_groups = list(pd.unique(group_values))
            full_data = source_data.copy()
            for group in unique_groups:
                mask = (group_values == group).values
                if not mask.any():
                    continue
                group_data = {key: np.asarray(values)[mask] for key, values in full_data.items()}
                group_source = ColumnDataSource(data=group_data)
                renderer = p.scatter(
                    x="x", y="y", source=group_source,
                    size="size", alpha=alpha, color=color_spec, marker="circle",
                    line_color="#333333", line_width=0.5,
                    selection_line_color="black", selection_line_width=2,
                )
                scatter_renderers.append(renderer)
                self._sources.append(group_source)
                self._source_ids.append(
                    [str(v) for v in group_source.data.get(id_col, [])]
                )
        else:
            source = ColumnDataSource(data=source_data)
            renderer = p.scatter(
                x="x", y="y", source=source,
                size="size", alpha=alpha, color=color_spec, marker="circle",
                line_color="#333333", line_width=0.5,
                selection_line_color="black", selection_line_width=2,
            )
            scatter_renderers.append(renderer)
            self._sources.append(source)
            self._source_ids.append(
                [str(v) for v in source.data.get(id_col, [])]
            )

        if hide_dots:
            for renderer in scatter_renderers:
                renderer.visible = False

        # Hover tool
        hover = HoverTool(
            tooltips=[
                (x_col, "@x{0.000}"),
                (y_col, "@y{0.000}"),
            ],
            renderers=scatter_renderers,
        )
        p.add_tools(hover)

        # Selection callback
        def on_select(_attr, _old, _new):
            if self._syncing_selection:
                return
            selected_ids = []
            for src, id_vals in zip(self._sources, self._source_ids):
                for idx in src.selected.indices or []:
                    if idx < len(id_vals):
                        selected_ids.append(str(id_vals[idx]))
            if selected_ids != self.data_holder.selected_record_ids:
                self.data_holder.selected_record_ids = selected_ids

        for src in self._sources:
            src.selected.on_change("indices", on_select)

        # Aggregation overlay
        if aggr_on and aggr_method != "---":
            self._render_aggr_on_cell(
                p, df_valid, x_col, y_col, group_column, color_mapper, groups,
                aggr_method, aggr_quantiles, aggr_n_quantiles, aggr_smooth,
                aggr_line_width,
            )

        # Style axes
        if not show_x_label:
            p.xaxis.axis_label = ""
            p.xaxis.major_label_text_font_size = "0pt"
        else:
            p.xaxis.axis_label = x_col
            p.xaxis.axis_label_text_font_size = f"{font_size}pt"
            p.xaxis.major_label_text_font_size = f"{tick_size}pt"

        if not show_y_label:
            p.yaxis.axis_label = ""
            p.yaxis.major_label_text_font_size = "0pt"
        else:
            p.yaxis.axis_label = y_col
            p.yaxis.axis_label_text_font_size = f"{font_size}pt"
            p.yaxis.major_label_text_font_size = f"{tick_size}pt"

        # Hide legend on individual cells (too cluttered)
        if p.legend:
            for legend in p.legend:
                legend.visible = False

        p.min_border = 5
        return p

    def _render_aggr_on_cell(
        self, p, df_valid, x_col, y_col, group_column, color_mapper, groups,
        method, use_quantiles, n_quantiles, smooth_factor, line_width,
    ):
        """Render aggregation curves on a scatter cell."""
        x_numeric = pd.to_numeric(df_valid[x_col], errors="coerce").values
        y_numeric = pd.to_numeric(df_valid[y_col], errors="coerce").values

        if group_column and group_column in df_valid.columns:
            group_values = df_valid[group_column].fillna("N/A").astype(str)
            color_map = {}
            if isinstance(color_mapper, CategoricalColorMapper):
                color_map = dict(zip(color_mapper.factors, color_mapper.palette))

            for cat, color in groups:
                mask = (group_values == cat).values
                x_g = x_numeric[mask].astype(float)
                y_g = y_numeric[mask].astype(float)
                result = compute_aggregation(
                    x_g, y_g, method, smooth_factor, use_quantiles, n_quantiles,
                )
                if not result:
                    continue
                line_kwargs = {"line_color": color, "line_width": line_width, "line_alpha": 0.9}
                if "p" in result and "r2" in result:
                    line_kwargs["line_dash"] = "solid" if result["p"] < 0.05 else "dashed"
                    stars = "***" if result["p"] < 0.001 else "**" if result["p"] < 0.01 else "*" if result["p"] < 0.05 else ""
                    line_kwargs["legend_label"] = f"R²={result['r2']:.2f}{stars}"
                p.line(result["x"], result["y"], **line_kwargs)
                if "y_upper" in result and "y_lower" in result:
                    band_source = ColumnDataSource(
                        data={"x": result["x"], "upper": result["y_upper"], "lower": result["y_lower"]}
                    )
                    band = Band(
                        base="x", upper="upper", lower="lower", source=band_source,
                        fill_color=color, fill_alpha=0.2, line_color=color, line_alpha=0.3,
                    )
                    p.add_layout(band)
        else:
            result = compute_aggregation(
                x_numeric.astype(float), y_numeric.astype(float),
                method, smooth_factor, use_quantiles, n_quantiles,
            )
            if result:
                line_kwargs = {"line_color": "black", "line_width": line_width, "line_alpha": 0.9}
                if "p" in result and "r2" in result:
                    line_kwargs["line_dash"] = "solid" if result["p"] < 0.05 else "dashed"
                    stars = "***" if result["p"] < 0.001 else "**" if result["p"] < 0.01 else "*" if result["p"] < 0.05 else ""
                    line_kwargs["legend_label"] = f"R²={result['r2']:.2f}{stars}"
                p.line(result["x"], result["y"], **line_kwargs)
                if "y_upper" in result and "y_lower" in result:
                    band_source = ColumnDataSource(
                        data={"x": result["x"], "upper": result["y_upper"], "lower": result["y_lower"]}
                    )
                    band = Band(
                        base="x", upper="upper", lower="lower", source=band_source,
                        fill_color="black", fill_alpha=0.15, line_color="black", line_alpha=0.3,
                    )
                    p.add_layout(band)

    def _render_plot(
        self,
        df: pd.DataFrame,
        columns: list[str],
        color_col: str,
        group_col: str,
        group_applies_to: str,
        size_col: str,
        palette: str,
        reverse_colors: bool,
        alpha: float,
        total_width: int,
        font_size: int,
        size_range: tuple[int, int],
        size_gamma: float,
        uniform_size: int,
        aggr_on: bool,
        aggr_method: str,
        aggr_quantiles: bool,
        aggr_n_quantiles: int,
        aggr_smooth: int,
        aggr_line_width: float,
        hist_bins: int,
        hist_kde: bool,
        hide_dots: bool,
        hide_color_bar: bool,
        hide_legend: bool,
    ):
        """Render the pair plot with current settings."""
        try:
            self._update_column_options(df)

            columns = list(self.columns_select.value or columns)
            color_col = self.color_select.value or color_col
            group_col = self.group_select.value or group_col
            size_col = self.size_select.value or size_col

            # Apply group override for color
            has_group = group_col not in ("---", None, "")
            if has_group:
                color_col = group_col

            if not columns or len(columns) < 2:
                return pn.pane.Markdown(
                    "Select at least 2 numeric columns.",
                    css_classes=["alert", "alert-info", "p-3"],
                )

            # Reset sources for fresh render
            self._sources = []
            self._source_ids = []

            return self._create_pair_grid(
                df, columns, color_col, group_col, size_col,
                palette, reverse_colors, alpha, int(total_width),
                int(font_size), size_range, size_gamma, uniform_size,
                aggr_on, aggr_method, aggr_quantiles, aggr_n_quantiles,
                aggr_smooth, aggr_line_width,
                int(hist_bins), hist_kde, hide_dots, hide_color_bar, hide_legend,
            )
        except Exception as e:
            logger.error(f"Error rendering pair plot: {e}")
            return pn.pane.Markdown(
                f"Error rendering pair plot: {e}",
                css_classes=["alert", "alert-danger", "p-3"],
            )

    def create(self) -> pn.viewable.Viewable:
        """Create the pair plot component with controls."""
        controls = pn.Column(
            self.clear_selection_button,
            self.columns_select,
            pn.layout.Divider(),
            self.color_select,
            self.palette_select,
            self.reverse_colors_toggle,
            pn.layout.Divider(),
            self.size_select,
            self.size_uniform_slider,
            self.size_range_slider,
            self.size_gamma_slider,
            pn.layout.Divider(),
            self.group_select,
            self.group_applies_to,
            pn.layout.Divider(),
            self.aggr_toggle,
            self.aggr_method,
            self.aggr_quantiles,
            self.aggr_n_quantiles,
            self.aggr_smooth,
            self.aggr_line_width_slider,
            pn.layout.Divider(),
            self.hist_bins_slider,
            self.hist_kde_toggle,
            pn.layout.Divider(),
            pn.Card(
                self.hide_dots,
                self.hide_color_bar,
                self.hide_legend,
                self.alpha_slider,
                self.width_slider,
                self.font_size_slider,
                title="More settings",
                collapsed=True,
                sizing_mode="stretch_width",
            ),
            width=200,
        )

        self._sync_url_state()

        plot = pn.bind(
            self._render_plot,
            df=self.data_holder.param.filtered_df,
            columns=self.columns_select,
            color_col=self.color_select,
            group_col=self.group_select,
            group_applies_to=self.group_applies_to,
            size_col=self.size_select,
            palette=self.palette_select,
            reverse_colors=self.reverse_colors_toggle,
            alpha=self.alpha_slider,
            total_width=self.width_slider,
            font_size=self.font_size_slider,
            size_range=self.size_range_slider,
            size_gamma=self.size_gamma_slider,
            uniform_size=self.size_uniform_slider,
            aggr_on=self.aggr_toggle,
            aggr_method=self.aggr_method,
            aggr_quantiles=self.aggr_quantiles,
            aggr_n_quantiles=self.aggr_n_quantiles,
            aggr_smooth=self.aggr_smooth,
            aggr_line_width=self.aggr_line_width_slider,
            hist_bins=self.hist_bins_slider,
            hist_kde=self.hist_kde_toggle,
            hide_dots=self.hide_dots,
            hide_color_bar=self.hide_color_bar,
            hide_legend=self.hide_legend,
        )

        return pn.Row(
            controls,
            pn.Spacer(width=20),
            pn.Column(
                pn.pane.Markdown("### Pair Plot"),
                plot,
                sizing_mode="stretch_width",
            ),
            sizing_mode="stretch_width",
        )
