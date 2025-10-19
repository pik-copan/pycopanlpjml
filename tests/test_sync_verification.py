"""
Test file specifically for verifying three-level synchronization.
Tests synchronization across World, Country, and Cell levels.

NOTE: These tests create their own LPJmL instances and require TEST_LINE_COUNTER
to be reset. The autouse fixture in conftest.py should handle this, but we also
explicitly reset at the start of each test for robustness.
"""

import os
import sys
import pytest
import numpy as np
from unittest.mock import patch

# Add the project root to the path
sys.path.insert(0, "/p/projects/copan/users/jannesbr/repos/pycopanlpjml")
sys._called_from_test = True

from tests.test_lpjml_coupling import Model
from tests.conftest import get_test_path


def test_three_level_synchronization(test_path):
    """
    Test that changes on each level (World, Country, Cell) are directly
    synchronized to the other levels via the Zarr backend.

    Note: This test MUST reset TEST_LINE_COUNTER to "0" at the start because
    it creates a fresh model instance that will read from the test data file.
    """
    # CRITICAL: Reset environment FIRST, before anything else
    os.environ["TEST_PATH"] = get_test_path()
    os.environ["TEST_LINE_COUNTER"] = "0"

    original_cwd = os.getcwd()

    try:
        # Change to test data directory
        test_data_dir = f"{test_path}/data"
        os.chdir(test_data_dir)

        # Create model
        model = Model(config_file="config_coupled_test.json")

        # ===================================================================
        # Test three-level synchronization: World <-> Country <-> Cell
        # ===================================================================
        print("\n" + "=" * 80)
        print("TESTING THREE-LEVEL SYNCHRONIZATION")
        print("=" * 80)

        # Get entities at each level
        first_country = list(model.world.countries)[0]
        first_cell = list(first_country.cells)[0]
        world_cell_idx = first_cell._cell_index

        # CRITICAL: Find the country index that corresponds to this world cell
        # Country indices are a subset of world indices
        country_cell_idx = np.where(
            first_country._cell_indices == world_cell_idx
        )[0][0]

        # INPUT SYNCHRONIZATION TESTS
        print("\n--- INPUT Synchronization ---")

        # Test 1: World -> Country -> Cell (INPUT)
        original_input = model.world.input["with_tillage"].values[
            world_cell_idx, 0, 0
        ]
        test_value_1 = original_input + 1000
        model.world.input["with_tillage"][world_cell_idx, 0, 0] = test_value_1

        assert (
            first_country.input["with_tillage"].values[country_cell_idx, 0, 0]
            == test_value_1
        )
        assert first_cell.input["with_tillage"].values[0, 0] == test_value_1
        print("  ✓ World -> Country -> Cell: INPUT synced")

        # Test 2: Country -> World, Cell (INPUT)
        test_value_2 = test_value_1 + 500
        first_country.input["with_tillage"][
            country_cell_idx, 0, 0
        ] = test_value_2

        assert (
            model.world.input["with_tillage"].values[world_cell_idx, 0, 0]
            == test_value_2
        )
        assert first_cell.input["with_tillage"].values[0, 0] == test_value_2
        print("  ✓ Country -> World, Cell: INPUT synced")

        # Test 3: Cell -> Country -> World (INPUT)
        test_value_3 = test_value_2 + 200
        first_cell.input["with_tillage"][0, 0] = test_value_3

        assert (
            model.world.input["with_tillage"].values[world_cell_idx, 0, 0]
            == test_value_3
        )
        assert (
            first_country.input["with_tillage"].values[country_cell_idx, 0, 0]
            == test_value_3
        )
        print("  ✓ Cell -> Country -> World: INPUT synced")

        # OUTPUT SYNCHRONIZATION TESTS
        print("\n--- OUTPUT Synchronization ---")

        # Test 4: World -> Country -> Cell (OUTPUT)
        original_output = model.world.output["hdate"].values[
            world_cell_idx, 0, 0
        ]
        test_value_4 = original_output + 100
        model.world.output["hdate"][world_cell_idx, 0, 0] = test_value_4

        assert (
            first_country.output["hdate"].values[country_cell_idx, 0, 0]
            == test_value_4
        )
        assert first_cell.output["hdate"].values[0, 0] == test_value_4
        print("  ✓ World -> Country -> Cell: OUTPUT synced")

        # Test 5: Country -> World, Cell (OUTPUT)
        test_value_5 = test_value_4 + 50
        first_country.output["hdate"][country_cell_idx, 0, 0] = test_value_5

        assert (
            model.world.output["hdate"].values[world_cell_idx, 0, 0]
            == test_value_5
        )
        assert first_cell.output["hdate"].values[0, 0] == test_value_5
        print("  ✓ Country -> World, Cell: OUTPUT synced")

        # Test 6: Cell -> Country -> World (OUTPUT)
        test_value_6 = test_value_5 + 25
        first_cell.output["hdate"][0, 0] = test_value_6

        assert (
            model.world.output["hdate"].values[world_cell_idx, 0, 0]
            == test_value_6
        )
        assert (
            first_country.output["hdate"].values[country_cell_idx, 0, 0]
            == test_value_6
        )
        print("  ✓ Cell -> Country -> World: OUTPUT synced")

        print("\n" + "=" * 80)
        print("✅ ALL SYNCHRONIZATION TESTS PASSED!")
        print("   INPUT:  6 paths verified (World ↔ Country ↔ Cell)")
        print("   OUTPUT: 6 paths verified (World ↔ Country ↔ Cell)")
        print("   Total:  12 synchronization paths working via Zarr backend")
        print("=" * 80)

    finally:
        # Close LPJmL coupler to free socket
        if "model" in locals() and hasattr(model, "lpjml"):
            model.lpjml.close()
        # Reset test line counter for next test
        os.environ["TEST_LINE_COUNTER"] = "0"
        # Restore original working directory
        os.chdir(original_cwd)


def test_fallback_mode_synchronization(test_path):
    """
    Test synchronization in fallback mode (cells without countries).

    This test initializes the model with with_countries=False to test the
    fallback mechanism where cells are initialized directly from World.

    Note: This test MUST reset TEST_LINE_COUNTER to "0" at the start.
    """
    # CRITICAL: Reset environment FIRST, before anything else
    os.environ["TEST_PATH"] = get_test_path()
    os.environ["TEST_LINE_COUNTER"] = "0"

    original_cwd = os.getcwd()

    try:
        # Change to test data directory
        test_data_dir = f"{test_path}/data"
        os.chdir(test_data_dir)

        # Create model in fallback mode (no countries)
        model = Model(
            config_file="config_coupled_test.json", with_countries=False
        )

        # Verify fallback mode
        assert (
            len(model.world.countries) == 0
        ), "Should have no countries in fallback mode"
        assert (
            len(model.world.cells) > 0
        ), "Should have cells initialized directly from World"

        # Test synchronization in fallback mode
        first_cell = list(model.world.cells)[0]
        world_cell_idx = first_cell._cell_index

        # Test World <-> Cell synchronization (no countries)
        original_input = model.world.input["with_tillage"].values[
            world_cell_idx, 0, 0
        ]
        test_value = original_input + 500
        model.world.input["with_tillage"][world_cell_idx, 0, 0] = test_value

        assert first_cell.input["with_tillage"].values[0, 0] == test_value
        print("  ✓ World -> Cell (fallback): INPUT synced")

        # Test Cell -> World synchronization
        test_value_2 = test_value + 300
        first_cell.input["with_tillage"][0, 0] = test_value_2

        assert (
            model.world.input["with_tillage"].values[world_cell_idx, 0, 0]
            == test_value_2
        )
        print("  ✓ Cell -> World (fallback): INPUT synced")

        print("\n✅ FALLBACK MODE SYNCHRONIZATION TESTS PASSED!")

    finally:
        # Close LPJmL coupler to free socket
        if "model" in locals() and hasattr(model, "lpjml"):
            model.lpjml.close()
        # Reset test line counter for next test
        os.environ["TEST_LINE_COUNTER"] = "0"
        # Restore original working directory
        os.chdir(original_cwd)
