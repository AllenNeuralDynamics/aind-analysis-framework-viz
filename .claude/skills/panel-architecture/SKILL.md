---
name: panel-architecture
description: Best practices for building Panel visualization apps at AIND using param.Parameterized classes, reactive binding with pn.bind(), and component patterns. Use when building or refactoring Panel applications.
---

# Panel Application Architecture

This skill captures best practices for building Panel visualization apps at AIND.

## Core Patterns

### 1. Parameterized Classes with DataHolder

Use `param.Parameterized` for reactive state management:

```python
import param
import panel as pn
import pandas as pd

class DataHolder(param.Parameterized):
    """Central state container that components can watch."""

    selected_record_id = param.String(default="")
    filtered_df = param.DataFrame()

class MyApp(param.Parameterized):
    def __init__(self):
        self.data_holder = DataHolder()
        self.df_full = self.load_data()
        self.data_holder.filtered_df = self.df_full.copy()
```

**Why this pattern?**
- Loose coupling between components
- Automatic UI updates when parameters change
- Clean separation of state from presentation

### 2. Reactive Binding with pn.bind()

Connect UI updates to parameter changes:

```python
# Simple binding - function called when parameter changes
display = pn.bind(
    self.render_content,
    record_id=self.data_holder.param.selected_record_id,
    df=self.data_holder.param.filtered_df,
)

# In layout
pn.Column(display)
```

### 3. Component Pattern

Each visualization component should:
1. Accept `df_meta` and `data_holder` in `__init__`
2. Provide `create_plot_controls()` returning widget dict
3. Have an `update_*()` method bound to controls
4. Store figures in `_latest_figures` for export

```python
class ScatterPlot:
    def __init__(self, df_meta: pd.DataFrame, data_holder: DataHolder):
        self.df_meta = df_meta
        self.data_holder = data_holder
        self.controls = self.create_plot_controls()
        self._latest_figures = {}

    def create_plot_controls(self) -> dict:
        return {
            "x_axis": pn.widgets.Select(name="X Axis", options=[...]),
            "y_axis": pn.widgets.Select(name="Y Axis", options=[...]),
        }

    def update_plot(self, x_col, y_col, df_meta=None):
        # Create and return the plot
        ...
        self._latest_figures["scatter"] = fig
        return layout
```

### 4. URL State Synchronization (Two-Way Sync)

Persist UI state in URL for shareable links. `pn.state.location.sync()` automatically handles **both directions**:
- URL changes → widget updates (on page load with URL params)
- Widget changes → URL updates (as user interacts)

#### Centralized URL Sync Pattern

Organize all URL sync logic in a single method in your main app:

```python
def _sync_url_state(self, tabs, spike_controls, filter_query=None):
    """Centralize all URL sync logic for the app."""
    location = pn.state.location

    # Sync tab selection
    location.sync(tabs, {'active': 'tab'})

    # Sync data holder parameters (param.Parameterized class)
    location.sync(
        self.data_holder,
        {
            'ephys_roi_id_selected': 'cell_id',
            'sweep_number_selected': 'sweep',
        },
    )

    # Sync spike analysis controls using a mapping dict
    spike_mapping = {
        'extract_from': 'spike_extract',
        'dim_reduction_method': 'dim_method',
        'n_clusters': 'n_clusters',
        'alpha_slider': 'alpha',
    }
    for control_name, url_param in spike_mapping.items():
        location.sync(spike_controls[control_name], {'value': url_param})

    # Sync scatter plot controls via component method
    self.scatter_plot.sync_controls_to_url()

    # Sync text input
    if filter_query is not None:
        location.sync(filter_query, {'value': 'query'})
```

Call this method at the end of `main_layout()` after all widgets are created:
```python
def main_layout(self):
    # ... create all widgets and layouts ...
    tabs = pn.Tabs(...)
    self._sync_url_state(tabs, spike_controls, filter_query)
    return template
```

#### Component-Level URL Sync

Each component can have its own `sync_controls_to_url()` method:

```python
class ScatterPlot:
    def sync_controls_to_url(self):
        """Sync scatter plot controls to URL query parameters."""
        location = pn.state.location
        mapping = {
            "x_axis_select": ("value", "scatter_x"),
            "y_axis_select": ("value", "scatter_y"),
            "color_col_select": ("value", "scatter_color"),
            "size_range_slider": ("value", "scatter_size_range"),
            "alpha_slider": ("value", "scatter_alpha"),
            "show_gmm": ("value", "scatter_gmm"),
            "show_linear_fit": ("value", "scatter_linear_fit"),
        }
        for control_name, (param_name, url_param) in mapping.items():
            location.sync(self.controls[control_name], {param_name: url_param})
```

#### Widget-Specific Sync Patterns

| Widget Type | Parameter | Example |
|-------------|-----------|---------|
| **Tabs** | `active` | `location.sync(tabs, {'active': 'tab'})` |
| **Select** | `value` | `location.sync(select_widget, {'value': 'x_axis'})` |
| **IntSlider** | `value` | `location.sync(slider, {'value': 'n_clusters'})` |
| **FloatSlider** | `value` | `location.sync(slider, {'value': 'alpha'})` |
| **RangeSlider** | `value` | `location.sync(slider, {'value': 'size_range'})` |
| **Checkbox** | `value` | `location.sync(checkbox, {'value': 'show_gmm'})` |
| **TextAreaInput** | `value` | `location.sync(text_area, {'value': 'query'})` |
| **TextInput** | `value` | `location.sync(text_input, {'value': 'search'})` |
| **DataHolder param** | - | `location.sync(data_holder, {'param_name': 'url_key'})` |

#### DataHolder Parameter Sync

Sync `param.Parameterized` class parameters directly:

```python
class DataHolder(param.Parameterized):
    ephys_roi_id_selected = param.String(default="")
    sweep_number_selected = param.Integer(default=0)
    filtered_df = param.DataFrame()

# In main app
location.sync(
    self.data_holder,
    {
        'ephys_roi_id_selected': 'cell_id',    # param_name: url_key
        'sweep_number_selected': 'sweep',
    },
)
```

#### URL Parameter Best Practices

1. **Use descriptive URL keys**: `scatter_x` instead of `x`
2. **Prefix by component**: `scatter_alpha`, `spike_n_clusters`
3. **Keep keys short but readable**: `show_gmm` not `show_gaussian_mixture_model`
4. **Use consistent naming**: `_{widget_type}` suffix for related params

#### Example URL Result

```
https://myapp.com?tab=1&cell_id=12345&sweep=52&scatter_x=Y+(A+--+P)&scatter_y=ipfx_tau&scatter_color=injection+region&scatter_alpha=0.7&scatter_gmm=true
```

## Layout Patterns

### BootstrapTemplate

Standard template for AIND apps:

```python
template = pn.template.BootstrapTemplate(
    title="My Analysis Explorer",
    header_background="#0072B5",  # Allen Institute blue
    favicon="https://alleninstitute.org/wp-content/uploads/2021/10/cropped-favicon-32x32.png",
    main=[main_content],
    sidebar=[sidebar_content],
    theme="default",
)
template.sidebar_width = 220
```

### Tabulator for DataFrames

```python
table = pn.widgets.Tabulator(
    df,
    selectable=1,              # Single row selection
    disabled=True,             # Read-only
    frozen_columns=["id"],     # Keep columns visible when scrolling
    header_filters=True,       # Enable column filtering
    show_index=False,
    height=400,
    sizing_mode="stretch_width",
    groupby=["category"],      # Optional grouping
    stylesheets=[":host .tabulator {font-size: 12px;}"],
)

# Handle selection
def on_selection(event):
    if event.new:
        idx = event.new[0]
        record_id = df.iloc[idx]["_id"]
        data_holder.selected_record_id = str(record_id)

table.param.watch(on_selection, "selection")
```

### Global Filter Panel

```python
filter_query = pn.widgets.TextAreaInput(
    name="Query string",
    value="status == 'success'",
    placeholder="Enter pandas query",
    height=80,
)

filter_button = pn.widgets.Button(name="Apply", button_type="primary")
reset_button = pn.widgets.Button(name="Reset", button_type="light")
status = pn.pane.Markdown("")

def apply_filter(event):
    try:
        filtered = df_full.query(filter_query.value)
        data_holder.filtered_df = filtered
        status.object = f"Showing {len(filtered)} records"
    except Exception as e:
        status.object = f"Error: {e}"

filter_button.on_click(apply_filter)
```

## Bokeh Integration

### Scatter Plot with Hover Tooltips

```python
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool

# Create source from DataFrame
source = ColumnDataSource(df)

# Create figure
p = figure(
    x_axis_label="X",
    y_axis_label="Y",
    tools="pan,wheel_zoom,box_zoom,reset,tap",
    height=600,
    width=800,
)

# Add scatter
scatter = p.scatter(
    x="x_col",
    y="y_col",
    source=source,
    size=10,
    color="navy",
    alpha=0.6,
)

# Add hover with image tooltip
tooltips = """
<div style="border: 1px solid black; padding: 10px;">
    <b>ID:</b> @_id<br>
    <img src="@asset_url{safe}" width="400">
</div>
"""
hover = HoverTool(tooltips=tooltips, renderers=[scatter])
p.add_tools(hover)

# Handle tap selection
def on_tap(attr, old, new):
    if new:
        idx = new[0]
        record_id = df.iloc[idx]["_id"]
        data_holder.selected_record_id = str(record_id)

source.selected.on_change("indices", on_tap)
```

### Color Mapping

```python
from bokeh.models import CategoricalColorMapper, LinearColorMapper
from bokeh.palettes import Viridis256

# Categorical
mapper = CategoricalColorMapper(
    factors=["A", "B", "C"],
    palette=["red", "green", "blue"],
)
color = {"field": "category", "transform": mapper}

# Continuous
mapper = LinearColorMapper(
    palette=Viridis256,
    low=df["value"].quantile(0.01),
    high=df["value"].quantile(0.99),
)
color = {"field": "value", "transform": mapper}
```

## Best Practices

### Performance

1. **Use `pn.bind()` over callbacks** when possible - cleaner and more efficient
2. **Throttle updates** for expensive operations:
   ```python
   slider.param.value_throttled  # Only fires after user stops dragging
   ```
3. **Avoid recreating components** - update data sources instead

### Code Organization

```
code/
├── app.py              # Main app class and entry point
├── core/
│   ├── base_app.py     # Base class with common patterns
│   └── data_loader.py  # Data fetching logic
├── components/
│   ├── scatter_plot.py # Visualization components
│   ├── data_table.py   # Table display
│   └── filter_panel.py # Filter UI
└── utils/
    └── export.py       # SVG/figure export
```

### Error Handling

```python
def safe_render(func):
    """Decorator to catch errors in render functions."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return pn.pane.Markdown(
                f"Error: {e}",
                css_classes=["alert", "alert-danger"],
            )
    return wrapper
```

### Initialization

```python
# Initialize extensions once at module level
pn.extension("tabulator")

# Set document title
from bokeh.io import curdoc
curdoc().title = "My App"

# Create app and serve
app = MyApp()
layout = app.main_layout()
layout.servable()
```

## Running the App

```bash
# Development with auto-reload (recommended)
panel serve code/app.py --dev --show --port 5006

# Note: --dev enables auto-reload on file changes
# Install watchfiles for better experience: uv add watchfiles

# Production with specific origin
panel serve code/app.py --allow-websocket-origin=myhost.com --port 7860

# Docker
CMD ["panel", "serve", "code/app.py", "--address", "0.0.0.0", "--port", "7860", "--allow-websocket-origin=*"]
```

## Development with uv

```bash
# Create environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -e ".[dev]"

# Run app (two options)
uv run panel serve code/app.py --dev --show  # Without activating venv
panel serve code/app.py --dev --show          # With activated venv
```
