"""
Pytest configuration and fixtures for SSN tests.

This file provides shared fixtures that can be used by all test files
in the tests directory.
"""

import pytest
from pathlib import Path


def pytest_addoption(parser):
    """Add custom command-line options for pytest."""
    parser.addoption(
        "--config",
        action="store",
        default="config.test_mapSeed42_netSeed42.test_stim_FullFieldGrating_ori0p0_C1p00_dur3000.json",
        help="Path to SSN configuration JSON file"
    )
    parser.addoption(
        "--ssn-verbose",
        action="store_true",
        default=False,
        help="Enable verbose SSN output during tests"
    )


@pytest.fixture
def config_file(request):
    """
    Fixture that provides the path to the SSN configuration file.
    
    Can be overridden via command line:
        pytest --config=path/to/config.json
    
    Returns the absolute path to the config file.
    """
    config_path = request.config.getoption("--config")
    
    # If relative path, make it relative to the tests directory
    if not Path(config_path).is_absolute():
        config_path = Path(__file__).parent / config_path
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        pytest.fail(f"Config file not found: {config_path}")
    
    return str(config_path)


@pytest.fixture
def verbose(request):
    """
    Fixture that provides the verbose flag for SSN simulations.
    
    Can be enabled via command line:
        pytest --ssn-verbose
    
    Returns True if verbose output is enabled, False otherwise.
    """
    return request.config.getoption("--ssn-verbose")
