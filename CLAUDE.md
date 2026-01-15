# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A **generic, reusable Panel app** for exploring AIND (Allen Institute for Neural Dynamics) analysis results across multiple projects.

**Goal**: Create a modular framework that can be easily configured for different AIND projects, starting with dynamic foraging model fitting data.

**Scope**:
- Fetch records from MongoDB/DocumentDB (project-specific collections)
- Load corresponding assets from S3 (PNG figures, videos, tables)
- Display interactive dataframes with dynamic filtering
- Support ad-hoc visualizations (scatter plots, histograms)
- Show assets in hover tooltips or grid views

## Starting Requirements

**First Prototype** (dynamic-foraging-model-fitting):
- Collection: `"dynamic-foraging-model-fitting"`
- Data Access: Use `aind-analysis-arch-result-access` package
- Asset Column: `S3_location` (contains S3 paths for each record)
- Asset Types: Start with PNG figures only
- Authentication: Open access (no auth required for DocDB or S3)

## Commands

```bash
# Development (auto-reload on file changes)
panel serve code/app.py --dev --show

# Production (Code Ocean capsule)
python code/run_capsule.py
```

## Architecture

### Data Flow
1. **MongoDB (DocumentDB)** → `aind-analysis-arch-result-access` → DataFrame
2. **DataFrame** → `DataHolder` (param.Parameterized) → reactive UI updates
3. **S3 assets** → HTTPS URLs → displayed in browser

### Key Components (Planned)

- `code/app.py` - Main entry point with project-specific app class
- `code/config.py` - Project configuration (collection name, columns, color mappings)
- `code/core/base_app.py` - BaseApp class with DataHolder pattern
- `code/core/data_loader.py` - MongoDB + S3 loading abstraction
- `code/components/base.py` - BaseComponent ABC for all UI components
- `code/components/scatter_plot.py` - Generic scatter plot
- `code/components/data_table.py` - Tabulator wrapper
- `code/components/filter_panel.py` - Global pandas query filter
- `code/components/asset_viewer.py` - S3 image/video display
- `code/utils/` - Color/size mapping, export utilities

### State Management Pattern

```python
class DataHolder(param.Parameterized):
    selected_record_ids = param.List(default=[])
    filtered_df = param.DataFrame()

# Components bind to DataHolder parameters
pn.bind(render_func, df=data_holder.param.filtered_df)
```

### Two-Level Filtering
1. **DocDB Query** (JSON) - Filters at database level before loading
2. **Pandas Query** - Filters loaded DataFrame in memory

## MVP Features

### 1. URL State Synchronization
- Shareable links with filtered/sorted state
- Uses `pn.state.location.sync()` to bind parameters to URL query string
- Allows deep-linking to specific views

### 2. Dynamic Filtering
- Global filter panel with pandas query syntax
- Real-time filtering as user types
- Filters apply to all visualizations (table, scatter plots)

### 3. Scatter Plot Visualization
- Generic scatter plot component
- Configurable x/y columns
- Color mapping by categorical column
- Size mapping by numeric column
- Hover tooltips with S3 image preview

### 4. Asset Viewer
- Display PNG figures from S3
- Show in hover tooltips or grid view
- Construct URLs from `S3_location` column

## AIND Data Infrastructure

See `.claude/skills/aind-infrastructure.md` for:
- MongoDB/DocumentDB connection patterns
- S3 bucket structure (`aind-data-bucket` patterns)
- One-collection-per-project pattern
- How to use `aind-analysis-arch-result-access` package

**Reference**: https://github.com/AllenNeuralDynamics/aind-analysis-arch-result-access
- See `src/aind_analysis_arch_result_access/df_mle_model_fitting.py` for example data access pattern

## Panel Patterns

See `.claude/skills/panel-architecture.md` for:
- `param.Parameterized` reactive state
- DataHolder pattern for cross-component state
- `pn.bind()` for reactive updates
- URL sync with `pn.state.location`
- Bokeh integration patterns

## Design Principles

1. **Modularity**: Core framework should be project-agnostic; configure via `config.py`
2. **Reactivity**: Use `param.Parameterized` and `pn.bind()` for automatic UI updates
3. **Separation**: Separate data loading (backend) from visualization (frontend)
4. **Reusability**: Components should work with different DataFrames via configuration
5. **Skills First**: Document stable patterns in skills for future sessions

## Documentation Conventions

- Root `README.md` is for Hugging Face Space metadata and should link to `.github/README.md` for details
