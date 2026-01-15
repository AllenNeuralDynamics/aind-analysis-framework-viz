"""
Configuration module for AIND Analysis Framework Visualization App.

This package provides project-specific configuration settings, making the app
easily adaptable for different AIND projects.

Submodules:
    - models: Configuration dataclasses (AppConfig, AssetConfig, etc.)
    - projects: Project-specific configurations and registry

Note: DataLoader classes are in the `data` package.
"""

# Re-export loader classes from data package for backward compatibility
from data import (
    CACHE_MAX_ITEMS,
    CACHE_POLICY,
    CACHE_TTL,
    DataLoader,
    DynamicForagingDataLoader,
    GenericDataLoader,
)
from .models import (
    AppConfig,
    AssetConfig,
    DataTableConfig,
    FilterConfig,
    QueryConfig,
)
from .projects import (
    DYNAMIC_FORAGING_LIFETIME_CONFIG,
    DYNAMIC_FORAGING_MODEL_FITTING_CONFIG,
    DYNAMIC_FORAGING_NM_CONFIG,
    PROJECT_REGISTRY,
)

__all__ = [
    # Loaders
    "DataLoader",
    "DynamicForagingDataLoader",
    "GenericDataLoader",
    "CACHE_MAX_ITEMS",
    "CACHE_POLICY",
    "CACHE_TTL",
    # Models
    "AppConfig",
    "AssetConfig",
    "DataTableConfig",
    "FilterConfig",
    "QueryConfig",
    # Projects
    "PROJECT_REGISTRY",
    "DYNAMIC_FORAGING_MODEL_FITTING_CONFIG",
    "DYNAMIC_FORAGING_NM_CONFIG",
    "DYNAMIC_FORAGING_LIFETIME_CONFIG",
]
