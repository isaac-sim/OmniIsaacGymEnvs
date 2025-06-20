#!/bin/bash

# Simple test runner script to demonstrate testing infrastructure
echo "Testing Infrastructure Validation"
echo "================================="
echo ""

# Check if pyproject.toml exists
if [ -f "pyproject.toml" ]; then
    echo "✓ pyproject.toml found"
else
    echo "✗ pyproject.toml not found"
    exit 1
fi

# Check test directory structure
if [ -d "tests" ] && [ -d "tests/unit" ] && [ -d "tests/integration" ]; then
    echo "✓ Test directory structure created"
else
    echo "✗ Test directory structure incomplete"
    exit 1
fi

# Check for conftest.py
if [ -f "tests/conftest.py" ]; then
    echo "✓ conftest.py with fixtures found"
else
    echo "✗ conftest.py not found"
    exit 1
fi

# Check for validation tests
if [ -f "tests/unit/test_setup_validation.py" ] && [ -f "tests/integration/test_infrastructure_integration.py" ]; then
    echo "✓ Validation tests created"
else
    echo "✗ Validation tests not found"
    exit 1
fi

# Check .gitignore updates
if grep -q ".pytest_cache/" .gitignore && grep -q "coverage.xml" .gitignore && grep -q ".claude/\*" .gitignore; then
    echo "✓ .gitignore updated with testing entries"
else
    echo "✗ .gitignore not properly updated"
    exit 1
fi

echo ""
echo "All infrastructure components are in place!"
echo ""
echo "To use the testing infrastructure:"
echo "1. Install dependencies: poetry install"
echo "2. Run tests: poetry run test (or poetry run tests)"
echo "3. Run specific markers: poetry run pytest -m unit"
echo "4. Run without slow tests: poetry run pytest -m 'not slow'"
echo ""
echo "Coverage reports will be generated in:"
echo "- HTML: htmlcov/index.html"
echo "- XML: coverage.xml"