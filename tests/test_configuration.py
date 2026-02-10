"""Unit tests for configuration system."""

import os

import tempfile
import unittest
from unittest.mock import MagicMock, patch

import yaml

from pycopanlpjml.model import Model
from pycoupler.config import CoupledConfig, _from_yaml, read_yaml


class TestConfigurationLoading(unittest.TestCase):
    """Test configuration loading and validation."""

    def test_load_default_config(self):
        """Test loading default configuration."""
        default_config_path = os.path.join(
            os.path.dirname(__file__), "..", "pycopanlpjml", "config.yaml"
        )

        with open(default_config_path, "r") as f:
            config = yaml.safe_load(f)

        # Check LPJmL settings
        self.assertIn("lpjml_settings", config)
        self.assertTrue(config["lpjml_settings"]["country_code_to_name"])

        # Check parallelization settings
        self.assertIn("parallelization", config)
        self.assertEqual(config["parallelization"]["max_workers"], 0)
        self.assertFalse(config["parallelization"]["debug"])
        self.assertEqual(config["parallelization"]["mode"], "auto")

        # Check profiling defaults (disabled by default)
        self.assertIn("profiling", config)
        self.assertFalse(config["profiling"]["driver"])
        self.assertFalse(config["profiling"]["workers"])

    def test_load_test_config(self):
        """Test loading test configuration."""
        test_config_path = os.path.join(
            os.path.dirname(__file__), "data", "test_config.yaml"
        )

        with open(test_config_path, "r") as f:
            config = yaml.safe_load(f)

        # Check LPJmL settings
        self.assertTrue(config["lpjml_settings"]["country_code_to_name"])

        # Check parallelization settings
        self.assertEqual(config["parallelization"]["max_workers"], 4)
        self.assertTrue(config["parallelization"]["debug"])
        self.assertEqual(config["parallelization"]["mode"], "serial")

        # Check test settings
        self.assertIn("test_settings", config)
        self.assertTrue(config["test_settings"]["test_flag"])
        self.assertEqual(config["test_settings"]["test_number"], 42)
        self.assertEqual(config["test_settings"]["test_list"], [1, 2, 3])
        self.assertFalse(config["test_settings"]["nested"]["enabled"])

        # Check profiling overrides
        self.assertIn("profiling", config)
        self.assertFalse(config["profiling"]["driver"])
        self.assertTrue(config["profiling"]["workers"])

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid configuration
        valid_config = {
            "lpjml_settings": {"country_code_to_name": True},
            "parallelization": {
                "max_workers": 0,
                "debug": False,
                "mode": "auto",
            },
        }

        # Should not raise any exception
        yaml.dump(
            valid_config, tempfile.NamedTemporaryFile(mode="w", delete=False)
        )

    def test_config_invalid_mode(self):
        """Test configuration with invalid mode."""
        invalid_config = {"parallelization": {"mode": "invalid_mode"}}

        # This should be handled gracefully by the code
        # (not validated at YAML level, but handled in detection logic)
        yaml.dump(
            invalid_config, tempfile.NamedTemporaryFile(mode="w", delete=False)
        )


class TestConfigurationDefaults(unittest.TestCase):
    """Test configuration defaults and fallbacks."""

    def test_missing_parallelization_section(self):
        """Test configuration missing parallelization section."""
        config = {"lpjml_settings": {"country_code_to_name": True}}

        # Should use defaults for missing parallelization section
        from pycopanlpjml.parallelization import _get_parallelization_config

        parallel_config = _get_parallelization_config(config)

        self.assertEqual(parallel_config["max_workers"], 0)
        self.assertFalse(parallel_config["debug"])
        self.assertEqual(parallel_config["mode"], "auto")

    def test_partial_parallelization_config(self):
        """Test partial parallelization configuration."""
        config = {
            "parallelization": {
                "max_workers": 8
                # Missing debug and mode
            }
        }

        from pycopanlpjml.parallelization import _get_parallelization_config

        parallel_config = _get_parallelization_config(config)

        self.assertEqual(parallel_config["max_workers"], 8)
        self.assertFalse(parallel_config["debug"])  # Default
        self.assertEqual(parallel_config["mode"], "auto")  # Default

    def test_empty_config(self):
        """Test empty configuration."""
        config = {}

        from pycopanlpjml.parallelization import _get_parallelization_config

        parallel_config = _get_parallelization_config(config)

        # Should use all defaults
        self.assertEqual(parallel_config["max_workers"], 0)
        self.assertFalse(parallel_config["debug"])
        self.assertEqual(parallel_config["mode"], "auto")


class TestConfigurationTypes(unittest.TestCase):
    """Test configuration type handling."""

    def test_numeric_max_workers(self):
        """Test numeric max_workers values."""
        test_cases = [(0, 0), (1, 1), (32, 32), (128, 128)]

        for input_val, expected in test_cases:
            config = {"parallelization": {"max_workers": input_val}}

            from pycopanlpjml.parallelization import (
                _get_parallelization_config,
            )

            parallel_config = _get_parallelization_config(config)

            self.assertEqual(parallel_config["max_workers"], expected)

    def test_boolean_debug(self):
        """Test boolean debug values."""
        test_cases = [
            (True, True),
            (False, False),
            ("true", True),  # String boolean
            ("false", False),  # String boolean
        ]

        for input_val, expected in test_cases:
            config = {"parallelization": {"debug": input_val}}

            from pycopanlpjml.parallelization import (
                _get_parallelization_config,
            )

            parallel_config = _get_parallelization_config(config)

            self.assertEqual(parallel_config["debug"], expected)

    def test_string_mode(self):
        """Test string mode values."""
        test_cases = [
            ("auto", "auto"),
            ("mpi", "mpi"),
            ("serial", "serial"),
            ("AUTO", "AUTO"),  # Case sensitive
            ("invalid", "invalid"),  # Invalid modes should be preserved
        ]

        for input_val, expected in test_cases:
            config = {"parallelization": {"mode": input_val}}

            from pycopanlpjml.parallelization import (
                _get_parallelization_config,
            )

            parallel_config = _get_parallelization_config(config)

            self.assertEqual(parallel_config["mode"], expected)


class TestConfigurationIntegration(unittest.TestCase):
    """Test configuration integration with Model."""

    @patch("pycopanlpjml.model.LPJmLCoupler")
    def test_component_with_config(self, mock_lpjml_coupler):
        """Test Model initialization with configuration."""
        # Mock LPJmLCoupler
        mock_lpjml_instance = MagicMock()
        settings = mock_lpjml_instance.config.coupled_config.lpjml_settings
        settings.country_code_to_name = True
        mock_lpjml_coupler.return_value = mock_lpjml_instance

        # Test configuration
        test_config = {
            "parallelization": {
                "mode": "serial",
                "debug": True,
                "max_workers": 4,
            }
        }

        # Mock the config loading
        with patch("pycopanlpjml.model.read_yaml") as mock_read_yaml:
            mock_read_yaml.return_value = _from_yaml(
                test_config, CoupledConfig
            )

            component = Model(config_file="test_config.yaml")

            # Check that parallel executor was initialized with config
            self.assertIsNotNone(component._parallel_executor)
            self.assertEqual(
                component._parallel_executor.config.mode, "serial"
            )

    @patch("pycopanlpjml.model.LPJmLCoupler")
    def test_component_with_default_config(self, mock_lpjml_coupler):
        """Test Model initialization with default configuration."""
        # Mock LPJmLCoupler
        mock_lpjml_instance = MagicMock()
        settings = mock_lpjml_instance.config.coupled_config.lpjml_settings
        settings.country_code_to_name = True
        mock_lpjml_coupler.return_value = mock_lpjml_instance

        # Load default config
        default_config_path = os.path.join(
            os.path.dirname(__file__), "..", "pycopanlpjml", "config.yaml"
        )

        # Mock the config loading
        with patch("pycopanlpjml.model.read_yaml") as mock_read_yaml:
            # Return the actual default config when loading
            mock_read_yaml.side_effect = lambda path, cls: read_yaml(
                default_config_path, cls
            )

            component = Model(config_file="config.yaml")

            # Check that parallel executor was initialized
            self.assertIsNotNone(component._parallel_executor)
            # Mode should be auto (default)
            self.assertEqual(
                component._parallel_executor.config.mode, "serial"
            )  # Will be serial in test environment


class TestConfigurationFileHandling(unittest.TestCase):
    """Test configuration file handling."""

    def test_config_file_not_found(self):
        """Test handling of missing configuration file."""
        with self.assertRaises(FileNotFoundError):
            with open("nonexistent_config.yaml", "r") as f:
                yaml.safe_load(f)

    def test_config_file_invalid_yaml(self):
        """Test handling of invalid YAML."""
        invalid_yaml = """
        parallelization:
            max_workers: 4
            debug: true
            mode: 'serial'
        invalid_yaml: [unclosed list
        """

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(invalid_yaml)
            temp_file = f.name

        try:
            with self.assertRaises(yaml.YAMLError):
                with open(temp_file, "r") as f:
                    yaml.safe_load(f)
        finally:
            os.unlink(temp_file)

    def test_config_file_permissions(self):
        """Test handling of file permission issues."""
        # This test would require creating a file with no read permissions
        # which is platform-specific and complex to test reliably
        pass


class TestConfigurationEdgeCases(unittest.TestCase):
    """Test configuration edge cases."""

    def test_very_large_max_workers(self):
        """Test very large max_workers value."""
        config = {"parallelization": {"max_workers": 1000000}}

        from pycopanlpjml.parallelization import _get_parallelization_config

        parallel_config = _get_parallelization_config(config)

        self.assertEqual(parallel_config["max_workers"], 1000000)

    def test_negative_max_workers(self):
        """Test negative max_workers value."""
        config = {"parallelization": {"max_workers": -1}}

        from pycopanlpjml.parallelization import _get_parallelization_config

        parallel_config = _get_parallelization_config(config)

        self.assertEqual(parallel_config["max_workers"], -1)

    def test_none_values(self):
        """Test None values in configuration."""
        config = {
            "parallelization": {
                "max_workers": None,
                "debug": None,
                "mode": None,
            }
        }

        from pycopanlpjml.parallelization import _get_parallelization_config

        parallel_config = _get_parallelization_config(config)

        # None values should be replaced with defaults
        self.assertEqual(parallel_config["max_workers"], 0)
        self.assertFalse(parallel_config["debug"])
        self.assertEqual(parallel_config["mode"], "auto")


class TestCoupledConfigIntegration(unittest.TestCase):
    """Test integration with pycoupler's CoupledConfig."""

    def test_coupledconfig_creation(self):
        """Test creating CoupledConfig objects."""
        config_data = {
            "lpjml_settings": {"country_code_to_name": True},
            "parallelization": {
                "max_workers": 8,
                "debug": True,
                "mode": "mpi",
            },
        }

        config = _from_yaml(config_data, CoupledConfig)

        # Test access to nested config
        self.assertTrue(config.lpjml_settings.country_code_to_name)
        self.assertEqual(config.parallelization.max_workers, 8)
        self.assertTrue(config.parallelization.debug)
        self.assertEqual(config.parallelization.mode, "mpi")

    def test_coupledconfig_to_dict(self):
        """Test converting CoupledConfig to dictionary."""
        config_data = {
            "parallelization": {
                "max_workers": 4,
                "debug": False,
                "mode": "serial",
            }
        }

        config = CoupledConfig(config_data)
        config_dict = config.to_dict()

        self.assertIn("parallelization", config_dict)
        self.assertEqual(config_dict["parallelization"]["max_workers"], 4)
        self.assertFalse(config_dict["parallelization"]["debug"])
        self.assertEqual(config_dict["parallelization"]["mode"], "serial")

    def test_parallelization_config_with_coupledconfig(self):
        """Test parallelization config handling with CoupledConfig."""
        from pycopanlpjml.parallelization import _get_parallelization_config

        config_data = {
            "parallelization": {
                "max_workers": 16,
                "debug": True,
                # mode missing - should use default
            }
        }

        config = CoupledConfig(config_data)
        parallel_config = _get_parallelization_config(config)

        self.assertEqual(parallel_config["max_workers"], 16)
        self.assertTrue(parallel_config["debug"])
        self.assertEqual(parallel_config["mode"], "auto")  # Default


if __name__ == "__main__":
    unittest.main()
