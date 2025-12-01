import sys
import os
import torch.nn as nn
from types import SimpleNamespace

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.motor_factory import get_motor
from src.engines.qca_engine import Aetheria_Motor
from src.engines.qca_engine_polar import PolarEngine

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3)

def test_motor_factory():
    print("🧪 Testing Motor Factory...")
    
    model = MockModel()
    import torch
    device = torch.device('cpu')
    
    # Test 1: Default (Cartesian)
    config = SimpleNamespace(ENGINE_TYPE='CARTESIAN', GRID_SIZE=64, D_STATE=8)
    motor = get_motor(config, model, device)
    assert isinstance(motor, Aetheria_Motor)
    print("✅ Cartesian Engine created successfully")
    
    # Test 2: Polar
    config = SimpleNamespace(ENGINE_TYPE='POLAR', GRID_SIZE=64, D_STATE=8)
    motor = get_motor(config, model, device)
    assert isinstance(motor, PolarEngine)
    print("✅ Polar Engine created successfully")
    
    # Test 3: Quantum (Fallback to Cartesian for now)
    config = SimpleNamespace(ENGINE_TYPE='QUANTUM', GRID_SIZE=64, D_STATE=8)
    motor = get_motor(config, model, device)
    # Currently falls back to Aetheria_Motor but logs warning
    assert isinstance(motor, Aetheria_Motor) 
    print("✅ Quantum Engine created (fallback) successfully")
    
    # Test 4: Unknown (Fallback to Cartesian)
    config = SimpleNamespace(ENGINE_TYPE='UNKNOWN', GRID_SIZE=64, D_STATE=8)
    motor = get_motor(config, model, device)
    assert isinstance(motor, Aetheria_Motor)
    print("✅ Unknown Engine fallback successfully")

if __name__ == "__main__":
    test_motor_factory()
