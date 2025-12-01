import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace
from src.engines.compute_backend import LocalBackend, MockQuantumBackend
from src.motor_factory import get_motor
from src.services.connection_manager import connection_manager

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(16, 16, 3, padding=1)
    def forward(self, x):
        return self.conv(x)

def test_local_backend():
    backend = LocalBackend(device='cpu')
    assert backend.get_device().type == 'cpu'
    status = backend.get_status()
    assert status['type'] == 'local'
    assert status['status'] == 'ready'

def test_mock_quantum_backend():
    backend = MockQuantumBackend(num_qubits=10)
    assert str(backend.get_device()) == 'cpu'
    status = backend.get_status()
    assert status['type'] == 'quantum_mock'
    assert status['qubits'] == 10
    
    result = backend.execute('run_circuit')
    assert isinstance(result, dict)

def test_connection_manager():
    cm = connection_manager
    assert cm.get_backend('local') is not None
    assert cm.get_backend('quantum_mock') is not None
    
    # Test switching
    cm.set_active_backend('quantum_mock')
    assert isinstance(cm.get_active_backend(), MockQuantumBackend)
    cm.set_active_backend('local')
    assert isinstance(cm.get_active_backend(), LocalBackend)

def test_motor_factory_local():
    config = SimpleNamespace(ENGINE_TYPE='CARTESIAN', BACKEND_TYPE='LOCAL', GRID_SIZE=32, D_STATE=16)
    model = SimpleModel()
    motor = get_motor(config, model, device='cpu')
    
    assert motor.__class__.__name__ == 'Aetheria_Motor'
    # Check if device is correct (LocalBackend passes device to engine)
    assert str(motor.device) == 'cpu'

def test_motor_factory_mock_quantum():
    config = SimpleNamespace(ENGINE_TYPE='CARTESIAN', BACKEND_TYPE='QUANTUM_MOCK', GRID_SIZE=32, D_STATE=16)
    model = SimpleModel()
    motor = get_motor(config, model, device='cpu') # Device arg should be ignored/overridden by backend
    
    assert motor.__class__.__name__ == 'Aetheria_Motor'
    # Mock backend returns 'cpu' as device for compatibility
    assert str(motor.device) == 'cpu'

if __name__ == "__main__":
    # Manual run if pytest not available
    test_local_backend()
    test_mock_quantum_backend()
    test_connection_manager()
    test_motor_factory_local()
    test_motor_factory_mock_quantum()
    print("All tests passed!")
