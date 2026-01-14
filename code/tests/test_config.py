"""
Tests for configuration module, focusing on DataLoader interface.

Run with:
    pytest code/tests/test_config.py -v
Or:
    python code/tests/test_config.py
"""

import sys
from pathlib import Path

# Add code directory to path for imports
code_dir = Path(__file__).parent.parent
sys.path.insert(0, str(code_dir))

import pytest
import pandas as pd
from unittest.mock import patch


class TestDataLoader:
    """Test the DataLoader abstract base class."""

    def test_dataloader_is_abstract(self):
        """Test that DataLoader cannot be instantiated directly."""
        from config import DataLoader

        with pytest.raises(TypeError):
            DataLoader()  # type: ignore

    def test_dataloader_requires_load_method(self):
        """Test that DataLoader subclasses must implement load()."""
        from config import DataLoader

        # This should raise an error because load() is not implemented
        with pytest.raises(TypeError):

            class IncompleteLoader(DataLoader):
                pass

            IncompleteLoader()


class TestDynamicForagingDataLoader:
    """Test the DynamicForagingDataLoader implementation."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        from config import DynamicForagingDataLoader

        loader = DynamicForagingDataLoader()

        assert loader.include_metrics is True
        assert loader.include_latent_variables is False
        assert loader.download_figures is False
        assert loader.paginate_settings == {"paginate": False}

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        from config import DynamicForagingDataLoader

        loader = DynamicForagingDataLoader(
            include_metrics=False,
            include_latent_variables=True,
            download_figures=True,
            paginate_settings={"paginate": True, "page_size": 100},
        )

        assert loader.include_metrics is False
        assert loader.include_latent_variables is True
        assert loader.download_figures is True
        assert loader.paginate_settings == {"paginate": True, "page_size": 100}

    @patch('aind_analysis_arch_result_access.get_mle_model_fitting')
    def test_load_calls_get_mle_model_fitting(self, mock_get_mle):
        """Test that load() calls get_mle_model_fitting with correct parameters."""
        from config import DynamicForagingDataLoader

        # Setup mock
        mock_df = pd.DataFrame({"_id": [1, 2], "subject_id": ["A", "B"]})
        mock_get_mle.return_value = mock_df

        # Create loader and load
        loader = DynamicForagingDataLoader(
            include_metrics=False,
            include_latent_variables=True,
            download_figures=False,
        )
        query = {"subject_id": "A"}
        result = loader.load(query)

        # Verify call
        mock_get_mle.assert_called_once_with(
            from_custom_query=query,
            if_include_metrics=False,
            if_include_latent_variables=True,
            if_download_figures=False,
            paginate_settings={"paginate": False},
        )
        pd.testing.assert_frame_equal(result, mock_df)

    @patch('aind_analysis_arch_result_access.get_mle_model_fitting')
    def test_load_with_custom_paginate_settings(self, mock_get_mle):
        """Test that load() passes custom paginate settings."""
        from config import DynamicForagingDataLoader

        mock_df = pd.DataFrame({"_id": [1]})
        mock_get_mle.return_value = mock_df

        loader = DynamicForagingDataLoader(
            paginate_settings={"paginate": True, "page_size": 50},
        )
        loader.load({})

        mock_get_mle.assert_called_once_with(
            from_custom_query={},
            if_include_metrics=True,
            if_include_latent_variables=False,
            if_download_figures=False,
            paginate_settings={"paginate": True, "page_size": 50},
        )


class TestCustomDataLoader:
    """Test creating custom data loaders for other projects."""

    def test_custom_dataloader_implementation(self):
        """Test that a custom data loader can be created."""
        from config import DataLoader, AppConfig

        # Create a custom data loader for a hypothetical project
        class MockProjectDataLoader(DataLoader):
            def __init__(self, api_key: str, timeout: int = 30):
                self.api_key = api_key
                self.timeout = timeout

            def load(self, query: dict) -> pd.DataFrame:
                # Simulate loading from some API
                return pd.DataFrame({
                    "_id": [1, 2, 3],
                    "value": [10, 20, 30],
                    "query_used": [str(query)] * 3,
                })

        # Test that it works
        loader = MockProjectDataLoader(api_key="test-key", timeout=60)
        assert loader.api_key == "test-key"
        assert loader.timeout == 60

        result = loader.load({"test": "query"})
        assert len(result) == 3
        assert list(result.columns) == ["_id", "value", "query_used"]

        # Verify it can be used with AppConfig
        config = AppConfig(
            app_name="Mock Project",
            data_loader=loader,
        )
        assert isinstance(config.data_loader, DataLoader)
        assert isinstance(config.data_loader, MockProjectDataLoader)


class TestAppConfig:
    """Test the AppConfig class."""

    def test_default_config_has_dataloader(self):
        """Test that DEFAULT_CONFIG has a data loader."""
        from config import DEFAULT_CONFIG, DataLoader

        assert DEFAULT_CONFIG.data_loader is not None
        assert isinstance(DEFAULT_CONFIG.data_loader, DataLoader)

    def test_default_config_uses_dynamic_foraging_loader(self):
        """Test that DEFAULT_CONFIG uses DynamicForagingDataLoader."""
        from config import DEFAULT_CONFIG, DynamicForagingDataLoader

        assert isinstance(DEFAULT_CONFIG.data_loader, DynamicForagingDataLoader)

    def test_config_with_custom_loader(self):
        """Test creating AppConfig with a custom loader."""
        from config import DataLoader, AppConfig

        class CustomLoader(DataLoader):
            def load(self, query: dict) -> pd.DataFrame:
                return pd.DataFrame({"test": [1]})

        config = AppConfig(
            app_name="Test App",
            data_loader=CustomLoader(),
        )

        assert config.app_name == "Test App"
        assert isinstance(config.data_loader, CustomLoader)

    def test_config_loads_data_through_loader(self):
        """Test that config can load data through its data_loader."""
        from config import DEFAULT_CONFIG

        # This should work without errors
        query = DEFAULT_CONFIG.query.get_default_query()
        # Note: This will make actual API calls in real environment
        # In tests, you might want to mock this
        assert isinstance(query, dict)
        assert "$or" in query


def run_tests():
    """Run tests without pytest."""
    print("Running DataLoader tests...")
    TestDataLoader().test_dataloader_is_abstract()
    TestDynamicForagingDataLoader().test_init_with_defaults()
    TestDynamicForagingDataLoader().test_init_with_custom_params()
    TestCustomDataLoader().test_custom_dataloader_implementation()
    TestAppConfig().test_default_config_has_dataloader()
    TestAppConfig().test_default_config_uses_dynamic_foraging_loader()
    TestAppConfig().test_config_with_custom_loader()
    TestAppConfig().test_config_loads_data_through_loader()
    print("\nAll tests passed!")


if __name__ == "__main__":
    run_tests()
