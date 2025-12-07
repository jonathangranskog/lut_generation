# LUT Generation Tests

This directory contains the test suite for the LUT generation project.

## Running Tests

### Run All Tests

```bash
pytest
```

### Run Specific Test Files

```bash
pytest test/test_lut.py
pytest test/test_smoothness.py
```

### Run Specific Test Classes or Functions

```bash
pytest test/test_lut.py::TestLUTIO
pytest test/test_lut.py::TestLUTIO::test_write_and_read_roundtrip
```

### Run Tests by Marker

```bash
# Run only fast unit tests (excludes integration tests)
pytest -m "not integration"

# Run only integration tests
pytest -m integration

# Run only slow tests
pytest -m slow
```

### Verbose Output

```bash
pytest -v
pytest -vv  # Extra verbose
```

### See Print Statements

```bash
pytest -s
```

### Stop on First Failure

```bash
pytest -x
```

### Run Last Failed Tests

```bash
pytest --lf
```

## Test Organization

### Unit Tests
- `test_lut.py` - LUT I/O, application, and creation
- `test_smoothness.py` - Smoothness loss computation
- `test_grayscale.py` - Grayscale LUT functionality
- `test_vlm.py` - VLM loss functionality
- `test_sds.py` - SDS loss functionality

### Integration Tests
- `test_integration.py` - End-to-end workflow tests
  - Tests full `optimize` command
  - Tests full `infer` command
  - Tests complete pipeline (optimize → infer)

### Shared Fixtures
- `conftest.py` - Pytest fixtures shared across all tests
  - Temporary directories and files
  - Sample images and LUTs
  - Test data generators

## Test Markers

Tests are marked with the following pytest markers:

- `@pytest.mark.integration` - Integration tests (slower, test full workflows)
- `@pytest.mark.slow` - Slow tests (>1 second)
- `@pytest.mark.unit` - Fast unit tests (default, no marker needed)

## Writing New Tests

### Use Fixtures

```python
def test_my_feature(gradient_image, identity_lut_16, temp_dir):
    # Use provided fixtures
    result = apply_lut(gradient_image, identity_lut_16)
    assert result.shape == gradient_image.shape
```

### Organize into Classes

```python
class TestMyFeature:
    """Tests for my feature."""

    def test_basic_functionality(self):
        # Test code here
        pass

    def test_edge_case(self):
        # Test code here
        pass
```

### Use Pytest Assertions

```python
# Prefer pytest's assert
assert x == y

# For torch tensors, use torch.testing
torch.testing.assert_close(tensor1, tensor2, rtol=1e-5, atol=1e-5)
```

### Mark Slow/Integration Tests

```python
@pytest.mark.slow
def test_expensive_operation():
    pass

@pytest.mark.integration
def test_full_workflow():
    pass
```

## Continuous Integration

When adding new features, ensure:
1. Add unit tests for core functionality
2. Add integration tests if the feature affects CLI commands
3. All tests pass before creating a pull request
4. Test coverage remains high (aim for >80%)

## Dependencies

Tests require:
- pytest
- torch
- PIL (Pillow)
- All project dependencies (from pyproject.toml)

Optional but recommended:
- pytest-cov (for coverage reports)
- pytest-xdist (for parallel test execution)

```bash
# Run tests in parallel with pytest-xdist
pytest -n auto
```
