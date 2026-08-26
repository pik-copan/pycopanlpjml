import os
import pytest


def get_test_path():
    """Fixture for the test path."""
    return os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def test_path():
    """Fixture for the test path."""
    return get_test_path()


def pytest_configure(config):
    import sys

    sys._called_from_test = True
    os.environ["TEST_PATH"] = os.path.dirname(os.path.abspath(__file__))
    os.environ["TEST_LINE_COUNTER"] = "0"


def pytest_unconfigure(config):
    import sys
    import shutil
    import glob

    del sys._called_from_test

    # Clean up any leftover temporary Zarr stores
    temp_stores = glob.glob(
        os.path.join(os.environ.get("TMPDIR", "/tmp"), "world_data_*.zarr")
    )
    # Also check user's temp directory
    temp_stores.extend(glob.glob("/p/tmp/jannesbr/world_data_*.zarr"))

    for store_path in temp_stores:
        try:
            shutil.rmtree(store_path)
        except Exception:
            pass  # Ignore errors during cleanup
