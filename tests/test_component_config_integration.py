"""Unit tests for Model integration with pycoupler configuration."""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import tempfile
import yaml

from pycopanlpjml.model import Model
from pycoupler.config import CoupledConfig, _from_yaml


class TestComponentConfigIntegration(unittest.TestCase):
    """Test Model integration with pycoupler configuration system."""

    @patch("pycopanlpjml.model.LPJmLCoupler")
    def test_component_loads_pycopanlpjml_config(self, mock_coupler):
        """Test that Model loads pycopanlpjml configuration."""
        # Mock LPJmL coupler
        mock_coupler_instance = MagicMock()
        mock_coupler_instance.config.coupled_config.lpjml_settings.country_code_to_name = (  # noqa
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
            with patch("pycopanlpjml.model.read_yaml") as mock_read_yaml:
                mock_read_yaml.return_value = _from_yaml(
                    config_data, CoupledConfig
                )

                component = Model(config_file=config_file)

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

    @patch("pycopanlpjml.model.LPJmLCoupler")
    def test_component_default_config_fallback(self, mock_coupler):
        """Test that Model falls back to default config when no file
        found."""
        # Mock LPJmL coupler
        mock_coupler_instance = MagicMock()
        mock_coupler_instance.config.coupled_config.lpjml_settings.country_code_to_name = (  # noqa
            True
        )
        mock_coupler.return_value = mock_coupler_instance

        # Mock config file that doesn't exist
        with patch("os.path.exists") as mock_exists:
            mock_exists.return_value = False

            component = Model(config_file="nonexistent.yaml")

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

    @patch("pycopanlpjml.model.LPJmLCoupler")
    def test_component_config_search_paths(self, mock_coupler):
        """Test that Model searches multiple paths for config."""
        # Mock LPJmL coupler
        mock_coupler_instance = MagicMock()
        mock_coupler_instance.config.coupled_config.lpjml_settings.country_code_to_name = (  # noqa
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

            component = Model(config_file="test.yaml")

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

    @patch("pycopanlpjml.model.LPJmLCoupler")
    def test_component_with_lpjml_instance(self, mock_coupler):
        """Test Model initialization with existing LPJmL instance."""
        # Mock LPJmL coupler
        mock_coupler_instance = MagicMock()
        mock_coupler_instance.config.coupled_config.lpjml_settings.country_code_to_name = (  # noqa
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

            component = Model(lpjml=mock_coupler_instance)

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

    @patch("pycopanlpjml.model.LPJmLCoupler")
    def test_config_file_not_found_handling(self, mock_coupler):
        """Test handling when config file is not found."""
        mock_coupler_instance = MagicMock()
        mock_coupler_instance.config.coupled_config.lpjml_settings.country_code_to_name = (  # noqa
            True
        )
        mock_coupler.return_value = mock_coupler_instance

        with patch("os.path.exists") as mock_exists:
            mock_exists.return_value = False

            component = Model(config_file="nonexistent.yaml")

            # Should not raise exception, should use defaults
            self.assertIsNotNone(component.pycopanlpjml_config)
            self.assertIsInstance(component.pycopanlpjml_config, CoupledConfig)

    @patch("pycopanlpjml.model.LPJmLCoupler")
    def test_config_file_invalid_yaml_handling(self, mock_coupler):
        """Test handling of invalid YAML files."""
        mock_coupler_instance = MagicMock()
        mock_coupler_instance.config.coupled_config.lpjml_settings.country_code_to_name = (  # noqa
            True
        )
        mock_coupler.return_value = mock_coupler_instance

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
                # Make os.path.exists return True for the temp file, but False
                # for others
                def exists_side_effect(path):
                    return path == temp_file

                mock_exists.side_effect = exists_side_effect

                with patch(
                    "pycopanlpjml.model.read_yaml"
                ) as mock_read_yaml:
                    # Make read_yaml raise for the invalid file but succeed for
                    # default
                    def read_yaml_side_effect(path, config_class):
                        if path == temp_file:
                            raise yaml.YAMLError("Invalid YAML")
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

                    # Should fall back to defaults (create Model with invalid
                    # YAML path; config dir has no valid file, so defaults used)
                    component = Model(config_file=temp_file)

                    self.assertIsNotNone(component.pycopanlpjml_config)
                    self.assertIsInstance(
                        component.pycopanlpjml_config, CoupledConfig
                    )

        finally:
            os.unlink(temp_file)


if __name__ == "__main__":
    unittest.main()
