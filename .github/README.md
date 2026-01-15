# AIND Analysis Framework Viz

> **Work in Progress** 🚧

A generic, reusable Panel app for exploring AIND (Allen Institute for Neural Dynamics) analysis results across multiple projects.

## Features

- 🔄 **Multi-project support** - Switch between different AIND projects via dropdown
- 📊 **Interactive data table** - Filter, sort, and select records with Tabulator
- 🖼️ **Asset viewer** - Display PNG figures from S3 based on selected records
- 🔗 **URL state sync** - Shareable links that preserve filters, selections, and view state
- 🧩 **Modular architecture** - Easy to extend for new projects via configuration

## Environment Setup

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync
```

## Quick Start

```bash
# Development (auto-reload on changes)
panel serve code/app.py --dev --show
```

## Project Structure

```
code/
├── app.py              # Main entry point
├── config/             # Project configurations
├── core/               # BaseApp, DataHolder pattern
├── components/         # UI components (table, filters, asset viewer)
└── data/               # Data loaders (DocDB + S3)
```

## Architecture

The app uses Panel for reactive UI with `param.Parameterized` state management. Data flows from DocumentDB/MongoDB through a `DataHolder` to reactive components via `pn.bind()`. New projects are added by defining configuration in `code/config/projects.py`.

---

🤖 **Credits** - Developed by Han Hou with Claude Code
