# SSN Tests

This directory contains pytest-based tests for the SSN (Stabilized Supralinear Network) model.

## Setup

Make sure pytest is installed in your environment:

```bash
pip install pytest
```

## Running Tests

### Run all tests
```bash
cd /home/scat-raid4/share/FlexibleFerrets/model_sandbox.git/SSN
pytest tests/
```

### Run with verbose output
```bash
pytest tests/ -v
```

### Run a specific test file
```bash
pytest tests/test_overflow_handling.py
```

### Run a specific test function
```bash
pytest tests/test_overflow_handling.py::test_normal_run
```

### Run with custom config file
```bash
pytest tests/ --config=/path/to/your/config.json
```

### Enable verbose SSN output during tests
```bash
pytest tests/ --ssn-verbose
```

### Show print statements during test execution
```bash
pytest tests/ -s
```

## Test Files

- **test_overflow_handling.py**: Tests for numerical stability and overflow handling
  - `test_normal_run`: Verifies normal SSN simulations complete without errors
  - `test_overflow_scenario`: Tests that extreme parameters are handled appropriately
  - `test_extreme_recurrent_weights`: Tests overflow handling with extreme connectivity
  - `test_event_threshold`: Tests event-based early termination of integration

## Configuration

Test configuration is managed through:
- **conftest.py**: Pytest fixtures for shared test setup (config file, verbose flag)
- **pytest.ini**: Pytest configuration (in parent SSN directory)

## Writing New Tests

1. Create a new file following the pattern `test_*.py`
2. Import necessary modules and the fixtures from conftest.py:
   ```python
   import pytest
   from SSN import SSN, NumericalInstabilityError
   
   def test_my_feature(config_file, verbose):
       # Your test code here
       ssn = SSN("my_model", verbose=verbose)
       ssn.load_config(config_file)
       # ... rest of your test
       assert condition, "failure message"
   ```

3. Use pytest assertions instead of returning True/False:
   - `assert condition, "message"`
   - `pytest.raises(ExceptionType)` for expected exceptions
   - `pytest.approx()` for floating point comparisons

## Continuous Integration

These tests can be easily integrated into CI/CD pipelines (GitHub Actions, GitLab CI, etc.):

```yaml
# Example GitHub Actions
- name: Run tests
  run: |
    pip install pytest
    pytest tests/ -v
```
