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
        mock_cfg.DEVICE = torch.device('cpu')
        yield mock_cfg, mock_state

# Mock dependencies
@pytest.fixture
def mock_deps():
    # Patch where it is defined, as it is imported locally
    with patch('src.utils.load_experiment_config') as mock_load_cfg, \
         patch('src.model_loader.load_model') as mock_load_model, \
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

def test_factory_integration_python_standard(mock_globals, mock_deps):
    mock_cfg, mock_state = mock_globals
    mock_ws = MagicMock()
    mock_state['websockets'] = {'test_ws_id': mock_ws}
    
    async def run_test():
        # Test "python" engine (Standard)
        await handle_load_experiment({
            'ws_id': 'test_ws_id',
            'experiment_name': 'test_exp', 
            'force_engine': 'python'
        })
    
    asyncio.run(run_test())
    
    assert 'motor' in mock_state
    motor = mock_state['motor']
    assert isinstance(motor, Aetheria_Motor)
    assert not isinstance(motor, PolarMotorWrapper)
    assert not isinstance(motor, HybridMotorWrapper)
    assert mock_state['motor_type'] == 'python'

def test_factory_integration_polar(mock_globals, mock_deps):
    mock_cfg, mock_state = mock_globals
    mock_ws = MagicMock()
    mock_state['websockets'] = {'test_ws_id': mock_ws}
    
    async def run_test():
        # Test "polar" engine
        await handle_load_experiment({
            'ws_id': 'test_ws_id',
            'experiment_name': 'test_exp', 
            'force_engine': 'polar'
        })
    
    asyncio.run(run_test())
    
    assert 'motor' in mock_state
    motor = mock_state['motor']
    # Should be PolarMotorWrapper
    assert isinstance(motor, PolarMotorWrapper)
    assert mock_state['motor_type'] == 'polar'

def test_factory_integration_quantum(mock_globals, mock_deps):
    mock_cfg, mock_state = mock_globals
    mock_ws = MagicMock()
    mock_state['websockets'] = {'test_ws_id': mock_ws}
    
    async def run_test():
        # Test "quantum" engine
        await handle_load_experiment({
            'ws_id': 'test_ws_id',
            'experiment_name': 'test_exp', 
            'force_engine': 'quantum'
        })
    
    asyncio.run(run_test())
    
    assert 'motor' in mock_state
    motor = mock_state['motor']
    # Should be HybridMotorWrapper
    assert isinstance(motor, HybridMotorWrapper)
    assert mock_state['motor_type'] == 'quantum'

def test_factory_integration_harmonic(mock_globals, mock_deps):
    mock_cfg, mock_state = mock_globals
    mock_ws = MagicMock()
    mock_state['websockets'] = {'test_ws_id': mock_ws}
    
    # Test Harmonic Engine (Real)
    async def run_test():
        await handle_load_experiment({
            'ws_id': 'test_ws_id',
            'experiment_name': 'test_exp', 
            'force_engine': 'harmonic'
        })
    
    asyncio.run(run_test())
    
    assert 'motor' in mock_state
    motor = mock_state['motor']
    # Should be HarmonicMotorWrapper
    from src.motor_factory import HarmonicMotorWrapper
    assert isinstance(motor, HarmonicMotorWrapper)
    assert hasattr(motor, 'evolve_internal_state')
    assert hasattr(motor, 'get_dense_state')

def test_factory_integration_lattice(mock_globals, mock_deps):
    mock_cfg, mock_state = mock_globals
    mock_ws = MagicMock()
    mock_state['websockets'] = {'test_ws_id': mock_ws}
    
    # Test Lattice Engine (Real)
    async def run_test():
        await handle_load_experiment({
            'ws_id': 'test_ws_id',
            'experiment_name': 'test_exp', 
            'force_engine': 'lattice'
        })
    
    asyncio.run(run_test())
    
    assert 'motor' in mock_state
    motor = mock_state['motor']
    # Should be LatticeMotorWrapper
    from src.motor_factory import LatticeMotorWrapper
    assert isinstance(motor, LatticeMotorWrapper)
    assert hasattr(motor, 'evolve_internal_state')
    assert hasattr(motor, 'get_dense_state')
