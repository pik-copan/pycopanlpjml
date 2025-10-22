"""Unit tests for Component integration with pycoupler configuration."""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import tempfile
import yaml

# Add the package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pycopanlpjml.component import Component
from pycoupler.config import CoupledConfig, _from_yaml


class TestComponentConfigIntegration(unittest.TestCase):
    """Test Component integration with pycoupler configuration system."""

    @patch("pycopanlpjml.component.LPJmLCoupler")
    def test_component_loads_pycopanlpjml_config(self, mock_coupler):
        """Test that Component loads pycopanlpjml configuration."""
        # Mock LPJmL coupler
        mock_coupler_instance = MagicMock()
        mock_coupler_instance.config.coupled_config.lpjml_settings.country_code_to_name = (
            True
        )
        mock_coupler.return_value = mock_coupler_instance

        # Create a temporary config file
        config_data = {
            "lpjml_settings": {"country_code_to_name": True},
            "parallelization": {
                "max_workers": 8,
                "debug": True,
                "mode": "mpi",
            },
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(config_data, f)
            config_file = f.name

        try:
            # Mock the config loading to return our test config
            with patch("pycopanlpjml.component.read_yaml") as mock_read_yaml:
                mock_read_yaml.return_value = _from_yaml(
                    config_data, CoupledConfig
                )

                component = Component(config_file=config_file)

                # Check that pycopanlpjml config was loaded
                self.assertIsNotNone(component.pycopanlpjml_config)
                self.assertIsInstance(
                    component.pycopanlpjml_config, CoupledConfig
                )

                # Check parallelization settings
                self.assertEqual(
                    component.pycopanlpjml_config.parallelization.max_workers,
                    8,
                )
                self.assertTrue(
                    component.pycopanlpjml_config.parallelization.debug
                )
                self.assertEqual(
                    component.pycopanlpjml_config.parallelization.mode, "mpi"
                )

                # Check that parallel executor was initialized with config
                self.assertIsNotNone(component._parallel_executor)

        finally:
            os.unlink(config_file)

    @patch("pycopanlpjml.component.LPJmLCoupler")
    def test_component_default_config_fallback(self, mock_coupler):
        """Test that Component falls back to default config when no file found."""
        # Mock LPJmL coupler
        mock_coupler_instance = MagicMock()
        mock_coupler_instance.config.coupled_config.lpjml_settings.country_code_to_name = (
            True
        )
        mock_coupler.return_value = mock_coupler_instance

        # Mock config file that doesn't exist
        with patch("os.path.exists") as mock_exists:
            mock_exists.return_value = False

            component = Component(config_file="nonexistent.yaml")

            # Check that default config was used
            self.assertIsNotNone(component.pycopanlpjml_config)
            self.assertIsInstance(component.pycopanlpjml_config, CoupledConfig)

            # Check default values
            self.assertEqual(
                component.pycopanlpjml_config.parallelization.max_workers, 0
            )
            self.assertFalse(
                component.pycopanlpjml_config.parallelization.debug
            )
            self.assertEqual(
                component.pycopanlpjml_config.parallelization.mode, "auto"
            )

    @patch("pycopanlpjml.component.LPJmLCoupler")
    def test_component_config_search_paths(self, mock_coupler):
        """Test that Component searches multiple paths for config."""
        # Mock LPJmL coupler
        mock_coupler_instance = MagicMock()
        mock_coupler_instance.config.coupled_config.lpjml_settings.country_code_to_name = (
            True
        )
        mock_coupler.return_value = mock_coupler_instance

        # Create config in package directory
        config_data = {"parallelization": {"mode": "serial", "debug": True}}

        package_config_path = os.path.join(
            os.path.dirname(__file__), "..", "pycopanlpjml", "config.yaml"
        )

        # Backup original config if it exists
        original_exists = os.path.exists(package_config_path)
        if original_exists:
            with open(package_config_path, "r") as f:
                original_content = f.read()

        try:
            # Write test config
            with open(package_config_path, "w") as f:
                yaml.dump(config_data, f)

            component = Component(config_file="test.yaml")

            # Check that config was loaded from package directory
            self.assertIsNotNone(component.pycopanlpjml_config)
            self.assertEqual(
                component.pycopanlpjml_config.parallelization.mode, "serial"
            )
            self.assertTrue(
                component.pycopanlpjml_config.parallelization.debug
            )

        finally:
            # Restore original config
            if original_exists:
                with open(package_config_path, "w") as f:
                    f.write(original_content)
            elif os.path.exists(package_config_path):
                os.unlink(package_config_path)

    @patch("pycopanlpjml.component.LPJmLCoupler")
    def test_component_with_lpjml_instance(self, mock_coupler):
        """Test Component initialization with existing LPJmL instance."""
        # Mock LPJmL coupler
        mock_coupler_instance = MagicMock()
        mock_coupler_instance.config.coupled_config.lpjml_settings.country_code_to_name = (
            True
        )

        # Create default config in package directory
        config_data = {"parallelization": {"mode": "auto", "max_workers": 4}}

        package_config_path = os.path.join(
            os.path.dirname(__file__), "..", "pycopanlpjml", "config.yaml"
        )

        # Backup original config if it exists
        original_exists = os.path.exists(package_config_path)
        if original_exists:
            with open(package_config_path, "r") as f:
                original_content = f.read()

        try:
            # Write test config
            with open(package_config_path, "w") as f:
                yaml.dump(config_data, f)

            component = Component(lpjml=mock_coupler_instance)

            # Check that config was loaded
            self.assertIsNotNone(component.pycopanlpjml_config)
            self.assertEqual(
                component.pycopanlpjml_config.parallelization.mode, "auto"
            )
            self.assertEqual(
                component.pycopanlpjml_config.parallelization.max_workers, 4
            )

            # Check that LPJmL instance was used
            self.assertEqual(component.lpjml, mock_coupler_instance)

        finally:
            # Restore original config
            if original_exists:
                with open(package_config_path, "w") as f:
                    f.write(original_content)
            elif os.path.exists(package_config_path):
                os.unlink(package_config_path)


class TestConfigFileHandling(unittest.TestCase):
    """Test configuration file handling."""

    def test_config_file_not_found_handling(self):
        """Test handling when config file is not found."""
        with patch("os.path.exists") as mock_exists:
            mock_exists.return_value = False

            # Should not raise exception, should use defaults
            config = Component._load_pycopanlpjml_config("nonexistent.yaml")

            self.assertIsNotNone(config)
            self.assertIsInstance(config, CoupledConfig)

    def test_config_file_invalid_yaml_handling(self):
        """Test handling of invalid YAML files."""
        invalid_yaml = """
        parallelization:
            max_workers: 4
            debug: true
        invalid_yaml: [unclosed list
        """

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(invalid_yaml)
            temp_file = f.name

        try:
            with patch("os.path.exists") as mock_exists:
                # Make os.path.exists return True for the temp file, but False for others
                def exists_side_effect(path):
                    return path == temp_file

                mock_exists.side_effect = exists_side_effect

                with patch(
                    "pycopanlpjml.component.read_yaml"
                ) as mock_read_yaml:
                    # Make read_yaml raise for the invalid file but succeed for default
                    def read_yaml_side_effect(path, config_class):
                        if path == temp_file:
                            raise yaml.YAMLError("Invalid YAML")
                        # Return actual default config
                        from pycopanlpjml.component import Component

                        return _from_yaml(
                            {
                                "lpjml_settings": {
                                    "country_code_to_name": True
                                },
                                "parallelization": {
                                    "max_workers": 0,
                                    "debug": False,
                                    "mode": "auto",
                                },
                            },
                            config_class,
                        )

                    mock_read_yaml.side_effect = read_yaml_side_effect

                    # Should fall back to defaults
                    config = Component._load_pycopanlpjml_config(temp_file)

                    self.assertIsNotNone(config)
                    self.assertIsInstance(config, CoupledConfig)

        finally:
            os.unlink(temp_file)


if __name__ == "__main__":
    unittest.main()
