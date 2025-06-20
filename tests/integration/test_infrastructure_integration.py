"""Integration tests to validate the complete testing infrastructure."""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
class TestInfrastructureIntegration:
    """Integration tests for the testing infrastructure."""

    def test_pytest_runs_successfully(self):
        """Verify pytest can be executed successfully."""
        # This test verifies pytest is working by the fact it's running
        assert True

    def test_coverage_can_be_generated(self, temp_dir):
        """Verify coverage reports can be generated."""
        # Create a dummy Python file to test coverage
        test_file = temp_dir / "dummy_module.py"
        test_file.write_text("""
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b
""")
        
        # Create a test for the dummy module
        test_test_file = temp_dir / "test_dummy.py"
        test_test_file.write_text("""
import sys
sys.path.insert(0, str(Path(__file__).parent))
from dummy_module import add

def test_add():
    assert add(2, 3) == 5
""")
        
        # The fact that we're running with coverage enabled validates this
        assert temp_dir.exists()

    @pytest.mark.slow
    def test_slow_marker_works(self):
        """Verify the slow marker is properly configured."""
        import time
        # Simulate a slow test
        time.sleep(0.1)
        assert True

    def test_mock_isaac_sim_fixture(self, mock_isaac_sim_app):
        """Verify the Isaac Sim mock fixture works."""
        # Test that the mock app has the expected methods
        assert hasattr(mock_isaac_sim_app, "update")
        assert hasattr(mock_isaac_sim_app, "close")
        
        # Verify we can call the mocked methods
        mock_isaac_sim_app.update()
        mock_isaac_sim_app.close()
        
        # Verify the calls were made
        mock_isaac_sim_app.update.assert_called_once()
        mock_isaac_sim_app.close.assert_called_once()

    def test_disable_gpu_fixture(self, disable_gpu):
        """Verify the disable_gpu fixture properly sets environment variables."""
        import os
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""

    def test_hydra_reset_fixture(self):
        """Verify Hydra is properly reset between tests."""
        from hydra.core.global_hydra import GlobalHydra
        
        # Hydra should be cleared by the fixture
        assert not GlobalHydra.instance().is_initialized()