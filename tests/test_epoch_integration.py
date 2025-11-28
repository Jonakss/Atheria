import pytest
import asyncio
import torch
import time
from unittest.mock import MagicMock, Mock, AsyncMock
from src.services.data_processing_service import DataProcessingService
from src.server.server_state import g_state

# Mock Motor class
class MockMotor:
    def __init__(self):
        self.state = Mock()
        # Create a dummy tensor: [Batch, Height, Width, Channels]
        # Random noise should trigger Epoch 1 (Quantum Soup)
        self.state.psi = torch.randn(1, 64, 64, 16)

@pytest.mark.asyncio
async def test_epoch_integration():
    # Setup queues
    state_queue = asyncio.Queue()
    broadcast_queue = asyncio.Queue()
    
    # Initialize service
    service = DataProcessingService(state_queue, broadcast_queue)
    # Mock VisualizationPipeline to avoid heavy computation/errors
    service.viz_pipeline = MagicMock()
    service.viz_pipeline.generate_frame = AsyncMock(return_value={'data': 'dummy'})
    
    # Mock g_state
    g_state['is_paused'] = False
    g_state['current_fps'] = 60.0
    g_state['current_epoch'] = -1 # Reset to verify update
    
    # Create dummy state data
    motor = MockMotor()
    state_data = {
        'motor_ref': motor,
        'roi': None,
        'viz_type': 'density',
        'step': 100
    }
    
    # Start service
    await service.start()
    
    try:
        # Feed state
        await state_queue.put(state_data)
        
        # Wait for processing (allow loop to run)
        # We wait for the broadcast queue to have an item
        try:
            message = await asyncio.wait_for(broadcast_queue.get(), timeout=2.0)
        except asyncio.TimeoutError:
            pytest.fail("Timeout waiting for broadcast message")
            
        # Verify Payload
        assert message['type'] == 'simulation_frame'
        payload = message['payload']
        
        # Check simulation_info
        assert 'simulation_info' in payload
        sim_info = payload['simulation_info']
        
        assert 'epoch' in sim_info
        assert isinstance(sim_info['epoch'], int)
        # Random noise should be Epoch 1 (Quantum Soup) or 0 (Void) depending on energy
        # But definitely not -1
        assert sim_info['epoch'] >= 0
        
        # Check g_state update
        assert g_state['current_epoch'] == sim_info['epoch']
        assert 'epoch_metrics' in g_state
        
        print(f"✅ Verified Epoch: {sim_info['epoch']}")
        print(f"✅ Verified Metrics: {g_state['epoch_metrics']}")
        
    finally:
        await service.stop()

if __name__ == "__main__":
    # Allow running directly
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(test_epoch_integration())
    print("Test finished")
