"""Shared pytest fixtures and configuration for omniisaacgymenvs tests."""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Generator, Dict, Any

import pytest
import yaml
from omegaconf import DictConfig, OmegaConf


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files.
    
    Yields:
        Path: Path to the temporary directory
    """
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def mock_config() -> DictConfig:
    """Create a mock Hydra configuration for testing.
    
    Returns:
        DictConfig: Basic configuration for testing
    """
    config = {
        "task": {
            "name": "TestTask",
            "env": {
                "numEnvs": 1,
                "envSpacing": 1.0,
                "clipObservations": 1000.0,
                "clipActions": 1.0,
            },
            "sim": {
                "dt": 0.0083,
                "use_gpu_pipeline": False,
                "gravity": [0.0, 0.0, -9.81],
                "physx": {
                    "worker_thread_count": 4,
                    "solver_type": 1,
                    "use_gpu": False,
                    "solver_position_iteration_count": 8,
                    "solver_velocity_iteration_count": 0,
                    "contact_offset": 0.02,
                    "rest_offset": 0.001,
                    "bounce_threshold_velocity": 0.2,
                    "max_depenetration_velocity": 10.0,
                    "default_buffer_size_multiplier": 5.0,
                },
            },
        },
        "train": {
            "ppo": {
                "learn": {
                    "learning_rate": 3e-4,
                    "lr_schedule": "linear",
                    "schedule_type": "standard",
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "truncate_grads": True,
                    "grad_norm": 1.0,
                    "entropy_coef": 0.0,
                    "value_loss_coef": 0.5,
                    "ppo_clipping_epsilon": 0.2,
                    "kl_threshold": 0.01,
                    "max_epochs": 10,
                    "minibatch_size": 128,
                    "mini_epochs": 8,
                    "critic_coef": 2,
                    "clip_value": True,
                    "use_smooth_clamp": True,
                },
                "model": {
                    "name": "continuous_a2c_logstd",
                },
                "network": {
                    "name": "actor_critic",
                    "separate": False,
                    "space": {
                        "continuous": {
                            "mu_activation": "None",
                            "sigma_activation": "None",
                            "mu_init": {
                                "name": "default",
                            },
                            "sigma_init": {
                                "name": "const_initializer",
                                "val": 0,
                            },
                            "fixed_sigma": True,
                        },
                    },
                    "mlp": {
                        "units": [256, 128, 64],
                        "activation": "elu",
                        "d2rl": False,
                        "initializer": {
                            "name": "default",
                        },
                        "regularizer": {
                            "name": "None",
                        },
                    },
                },
                "load_checkpoint": False,
            },
        },
        "test": False,
        "checkpoint": "",
        "num_envs": 1,
        "physics_engine": "physx",
        "pipeline": "cpu",
        "sim_device": "cpu",
        "rl_device": "cpu",
        "graphics_device_id": 0,
        "headless": True,
        "seed": 42,
    }
    return OmegaConf.create(config)


@pytest.fixture
def sample_yaml_config(temp_dir: Path) -> Path:
    """Create a sample YAML configuration file for testing.
    
    Args:
        temp_dir: Temporary directory fixture
        
    Returns:
        Path: Path to the created YAML file
    """
    config_data = {
        "task": {
            "name": "SampleTask",
            "env": {
                "numEnvs": 10,
            },
        },
        "train": {
            "params": {
                "seed": 123,
                "algo": {
                    "name": "ppo",
                },
            },
        },
    }
    
    yaml_path = temp_dir / "sample_config.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(config_data, f)
    
    return yaml_path


@pytest.fixture
def mock_env_config() -> Dict[str, Any]:
    """Create a mock environment configuration dictionary.
    
    Returns:
        Dict[str, Any]: Environment configuration for testing
    """
    return {
        "numEnvs": 1,
        "numObservations": 10,
        "numActions": 4,
        "envSpacing": 1.0,
        "episodeLength": 1000,
        "enableDebugVis": False,
        "clipObservations": 1000.0,
        "clipActions": 1.0,
        "controlFrequencyInv": 1,
    }


@pytest.fixture
def cleanup_on_exit():
    """Fixture to ensure cleanup of test artifacts after test execution."""
    yield
    # Add any global cleanup logic here if needed
    pass


@pytest.fixture(autouse=True)
def reset_hydra():
    """Reset Hydra's global state between tests to avoid conflicts."""
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


@pytest.fixture
def mock_isaac_sim_app(mocker):
    """Mock Isaac Sim application for tests that require it.
    
    Args:
        mocker: pytest-mock fixture
        
    Returns:
        Mock object representing Isaac Sim app
    """
    mock_app = mocker.MagicMock()
    mock_app.update = mocker.MagicMock()
    mock_app.close = mocker.MagicMock()
    
    # Mock the SimulationApp import
    mocker.patch("omni.isaac.kit.SimulationApp", return_value=mock_app)
    
    return mock_app


@pytest.fixture
def disable_gpu():
    """Fixture to ensure GPU is disabled for tests."""
    original_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    yield
    if original_cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible_devices
    else:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)