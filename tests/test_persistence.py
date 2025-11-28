import pytest
import asyncio
from unittest.mock import MagicMock, patch
from src.server.server_state import g_state
from src.server.server_handlers import handle_load_experiment

def test_active_experiment_persistence_sync():
    asyncio.run(test_active_experiment_persistence())

async def test_active_experiment_persistence():
    # Setup
    g_state['websockets'] = {}
    g_state['active_experiment'] = None
    
    # Mock dependencies
    # Note: server_handlers imports these from ..utils and .model_loader inside the function
    # So we need to patch where they are defined or where they are imported from if they were top-level.
    # Since they are local imports, we should patch the source modules.
    with patch('src.utils.load_experiment_config') as mock_load_config, \
         patch('src.utils.get_latest_checkpoint') as mock_get_checkpoint, \
         patch('src.model_loader.load_model') as mock_load_model, \
         patch('src.engines.qca_engine.Aetheria_Motor') as mock_motor_cls:
        
        # Configure mocks
        mock_config = MagicMock()
        mock_config.MODEL_PARAMS.d_state = 16
        mock_load_config.return_value = mock_config
        mock_get_checkpoint.return_value = "/path/to/checkpoint.pth"
        
        mock_model = MagicMock()
        mock_load_model.return_value = (mock_model, {})
        
        mock_motor = MagicMock()
        mock_motor_cls.return_value = mock_motor
        
        # Execute handler
        args = {
            'ws_id': 'test_ws',
            'experiment_name': 'test_experiment_v1'
        }
        await handle_load_experiment(args)
        
        # Verify
        assert g_state['active_experiment'] == 'test_experiment_v1', "active_experiment should be updated in g_state"
