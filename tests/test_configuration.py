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

        # Check profiling defaults (disabled by default)
        self.assertIn("profiling", config)
        self.assertFalse(config["profiling"])

        # Check output settings
        self.assertIn("output", config)
        self.assertIn("format", config["output"])

    def test_load_test_config(self):
        """Test loading test configuration."""
        test_config_path = os.path.join(
            os.path.dirname(__file__), "data", "test_config.yaml"
        )

        with open(test_config_path, "r") as f:
            config = yaml.safe_load(f)

        # Check LPJmL settings
        self.assertTrue(config["lpjml_settings"]["country_code_to_name"])

        # Check test settings
        self.assertIn("test_settings", config)
        self.assertTrue(config["test_settings"]["test_flag"])
        self.assertEqual(config["test_settings"]["test_number"], 42)
        self.assertEqual(config["test_settings"]["test_list"], [1, 2, 3])
        self.assertFalse(config["test_settings"]["nested"]["enabled"])

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid configuration
        valid_config = {
            "lpjml_settings": {"country_code_to_name": True},
            "profiling": False,
            "output": {"format": ["netcdf"]},
        }

        # Should not raise any exception
        yaml.dump(
            valid_config, tempfile.NamedTemporaryFile(mode="w", delete=False)
        )


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
        lpjml_settings:
            country_code_to_name: true
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


class TestCoupledConfigIntegration(unittest.TestCase):
    """Test integration with pycoupler's CoupledConfig."""

    def test_coupledconfig_creation(self):
        """Test creating CoupledConfig objects."""
        config_data = {
            "lpjml_settings": {"country_code_to_name": True},
            "profiling": False,
            "output": {"format": ["netcdf", "csv"]},
        }

        config = _from_yaml(config_data, CoupledConfig)

        # Test access to nested config
        self.assertTrue(config.lpjml_settings.country_code_to_name)

    def test_coupledconfig_to_dict(self):
        """Test converting CoupledConfig to dictionary."""
        config_data = {
            "lpjml_settings": {"country_code_to_name": False},
            "profiling": True,
        }

        config = CoupledConfig(config_data)
        config_dict = config.to_dict()

        self.assertIn("lpjml_settings", config_dict)
        self.assertFalse(config_dict["lpjml_settings"]["country_code_to_name"])
        self.assertTrue(config_dict["profiling"])


if __name__ == "__main__":
    unittest.main()
