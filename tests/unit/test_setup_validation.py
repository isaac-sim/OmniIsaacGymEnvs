"""Validation tests to ensure the testing infrastructure is properly set up."""

import os
import sys
from pathlib import Path

import pytest
from omegaconf import DictConfig


class TestSetupValidation:
    """Test class to validate the testing infrastructure setup."""

    def test_python_version(self):
        """Verify Python version is compatible."""
        assert sys.version_info >= (3, 7), "Python 3.7 or higher is required"

    def test_project_structure_exists(self):
        """Verify the basic project structure exists."""
        workspace_root = Path("/workspace")
        
        # Check main package exists
        assert (workspace_root / "omniisaacgymenvs").exists()
        assert (workspace_root / "omniisaacgymenvs" / "__init__.py").exists()
        
        # Check test directories exist
        assert (workspace_root / "tests").exists()
        assert (workspace_root / "tests" / "__init__.py").exists()
        assert (workspace_root / "tests" / "unit").exists()
        assert (workspace_root / "tests" / "unit" / "__init__.py").exists()
        assert (workspace_root / "tests" / "integration").exists()
        assert (workspace_root / "tests" / "integration" / "__init__.py").exists()
        
        # Check configuration files exist
        assert (workspace_root / "pyproject.toml").exists()
        assert (workspace_root / "tests" / "conftest.py").exists()

    def test_import_main_package(self):
        """Verify the main package can be imported."""
        try:
            import omniisaacgymenvs
            assert omniisaacgymenvs is not None
        except ImportError:
            pytest.fail("Failed to import omniisaacgymenvs package")

    @pytest.mark.unit
    def test_fixtures_available(self, temp_dir, mock_config, mock_env_config):
        """Verify pytest fixtures are available and working."""
        # Test temp_dir fixture
        assert temp_dir.exists()
        assert temp_dir.is_dir()
        
        # Test mock_config fixture
        assert isinstance(mock_config, DictConfig)
        assert "task" in mock_config
        assert "train" in mock_config
        
        # Test mock_env_config fixture
        assert isinstance(mock_env_config, dict)
        assert "numEnvs" in mock_env_config
        assert "numObservations" in mock_env_config

    @pytest.mark.unit
    def test_yaml_config_fixture(self, sample_yaml_config):
        """Verify the YAML config fixture works properly."""
        assert sample_yaml_config.exists()
        assert sample_yaml_config.suffix == ".yaml"
        
        # Verify we can read the file
        with open(sample_yaml_config, "r") as f:
            content = f.read()
            assert "SampleTask" in content
            assert "numEnvs" in content

    def test_coverage_configuration(self):
        """Verify coverage is properly configured."""
        workspace_root = Path("/workspace")
        pyproject_path = workspace_root / "pyproject.toml"
        
        with open(pyproject_path, "r") as f:
            content = f.read()
            
        # Check pytest coverage options
        assert "--cov=omniisaacgymenvs" in content
        assert "--cov-report=html:htmlcov" in content
        assert "--cov-report=xml:coverage.xml" in content
        assert "--cov-fail-under=80" in content
        
        # Check coverage configuration sections
        assert "[tool.coverage.run]" in content
        assert "[tool.coverage.report]" in content
        assert "[tool.coverage.html]" in content
        assert "[tool.coverage.xml]" in content

    @pytest.mark.integration
    def test_markers_configured(self, pytestconfig):
        """Verify custom pytest markers are configured."""
        markers = pytestconfig.getini("markers")
        marker_names = [m.split(":")[0] for m in markers]
        
        assert "unit" in marker_names
        assert "integration" in marker_names
        assert "slow" in marker_names

    def test_poetry_scripts_configured(self):
        """Verify Poetry scripts are properly configured."""
        workspace_root = Path("/workspace")
        pyproject_path = workspace_root / "pyproject.toml"
        
        with open(pyproject_path, "r") as f:
            content = f.read()
            
        # Check that both test and tests commands are configured
        assert '[tool.poetry.scripts]' in content
        assert 'test = "pytest:main"' in content
        assert 'tests = "pytest:main"' in content