---
title: AIND Analysis Framework Viz
emoji: 📈
colorFrom: gray
colorTo: green
sdk: docker
pinned: false
license: mit
short_description: 'Panel app for visualizing analysis results in AIND Analysis'
---

# AIND Analysis Framework Viz

A generic, reusable Panel app for exploring AIND (Allen Institute for Neural Dynamics) analysis results across multiple projects.

## Features

- **Multi-project support** - Switch between different AIND projects via dropdown
- **Interactive data table** - Filter, sort, and select records with Tabulator
- **Asset viewer** - Display PNG figures from S3 based on selected records
- **URL state sync** - Shareable links that preserve filters, selections, and view state
- **Modular architecture** - Easy to extend for new projects via configuration

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run with Panel
panel serve code/app.py --dev --show
```

For detailed documentation, see [.github/README.md](.github/README.md)
