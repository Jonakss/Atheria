import pytest
import torch
import asyncio
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

from src.motor_factory import get_motor, PolarMotorWrapper, HybridMotorWrapper
from src.engines.qca_engine import Aetheria_Motor
from src.pipelines.handlers.inference_handlers import handle_load_experiment

# Mock global config and state
@pytest.fixture
def mock_globals():
    with patch('src.pipelines.handlers.inference_handlers.global_cfg') as mock_cfg, \
         patch('src.pipelines.handlers.inference_handlers.g_state', {}) as mock_state:
        mock_cfg.GRID_SIZE_INFERENCE = 64
        mock_cfg.INITIAL_STATE_MODE_INFERENCE = 'complex_noise'
        yield mock_cfg, mock_state

# Mock dependencies
@pytest.fixture
def mock_deps():
    with patch('src.pipelines.handlers.inference_handlers.load_experiment_config') as mock_load_cfg, \
         patch('src.pipelines.handlers.inference_handlers.load_model') as mock_load_model, \
         patch('src.pipelines.handlers.inference_handlers.get_latest_checkpoint') as mock_get_ckpt, \
         patch('src.pipelines.handlers.inference_handlers.send_notification') as mock_notify:
        
        # Setup default config
        exp_cfg = SimpleNamespace(
            MODEL_PARAMS=SimpleNamespace(d_state=2),
            GRID_SIZE_TRAINING=64,
            GRID_SIZE_INFERENCE=64
        )
        mock_load_cfg.return_value = exp_cfg
        
        # Setup default model
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        
        yield mock_load_cfg, mock_load_model, mock_get_ckpt, mock_notify

@pytest.mark.asyncio
async def test_factory_integration_python_standard(mock_globals, mock_deps):
    mock_cfg, mock_state = mock_globals
    
    # Test "python" engine (Standard)
    await handle_load_experiment(
        ws=MagicMock(),
        data={'experiment_name': 'test_exp', 'force_engine': 'python'}
    )
    
    assert 'motor' in mock_state
    motor = mock_state['motor']
    assert isinstance(motor, Aetheria_Motor)
    assert not isinstance(motor, PolarMotorWrapper)
    assert not isinstance(motor, HybridMotorWrapper)
    assert mock_state['motor_type'] == 'python'

@pytest.mark.asyncio
async def test_factory_integration_polar(mock_globals, mock_deps):
    mock_cfg, mock_state = mock_globals
    
    # Test "polar" engine
    await handle_load_experiment(
        ws=MagicMock(),
        data={'experiment_name': 'test_exp', 'force_engine': 'polar'}
    )
    
    assert 'motor' in mock_state
    motor = mock_state['motor']
    # Should be PolarMotorWrapper
    assert isinstance(motor, PolarMotorWrapper)
    assert mock_state['motor_type'] == 'polar'

@pytest.mark.asyncio
async def test_factory_integration_quantum(mock_globals, mock_deps):
    mock_cfg, mock_state = mock_globals
    
    # Test "quantum" engine
    await handle_load_experiment(
        ws=MagicMock(),
        data={'experiment_name': 'test_exp', 'force_engine': 'quantum'}
    )
    
    assert 'motor' in mock_state
    motor = mock_state['motor']
    # Should be HybridMotorWrapper
    assert isinstance(motor, HybridMotorWrapper)
    assert mock_state['motor_type'] == 'quantum'

@pytest.mark.asyncio
async def test_factory_integration_harmonic(mock_globals, mock_deps):
    mock_cfg, mock_state = mock_globals
    
    # Test "harmonic" engine (handled specially in handler, not factory yet)
    # We need to mock SparseHarmonicEngine since it might not be importable if deps missing
    with patch('src.pipelines.handlers.inference_handlers.SparseHarmonicEngine') as MockHarmonic:
        await handle_load_experiment(
            ws=MagicMock(),
            data={'experiment_name': 'test_exp', 'force_engine': 'harmonic'}
        )
        
        assert 'motor' in mock_state
        assert mock_state['motor_type'] == 'harmonic'
        MockHarmonic.assert_called_once()
