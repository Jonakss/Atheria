import torch
import pytest
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.engines.lattice_engine import LatticeEngine

def test_lattice_initialization():
    grid_size = 16
    engine = LatticeEngine(grid_size=grid_size, d_state=9, device='cpu')
    
    # Check shape: [1, 2, H, W, 3, 3]
    assert engine.links.shape == (1, 2, grid_size, grid_size, 3, 3)
    
    # Check unitarity: U * U^dag = I
    U = engine.links
    U_dag = U.conj().transpose(-2, -1)
    prod = torch.matmul(U, U_dag)
    identity = torch.eye(3, dtype=torch.complex64).view(1, 1, 1, 1, 3, 3).expand_as(prod)
    
    # Allow small numerical error due to matrix_exp approximation or float precision
    assert torch.allclose(prod, identity, atol=1e-5), "Links must be unitary"

def test_wilson_action():
    grid_size = 8
    engine = LatticeEngine(grid_size=grid_size, d_state=9, device='cpu', beta=6.0)
    
    action = engine._compute_action(engine.links)
    assert isinstance(action.item(), float)
    assert action.item() < 0, "Action should be negative (convention S = -beta/N * ReTrP)"

def test_step_update():
    grid_size = 8
    engine = LatticeEngine(grid_size=grid_size, d_state=9, device='cpu')
    
    initial_links = engine.links.clone()
    engine.step()
    
    # Links should change
    assert not torch.allclose(initial_links, engine.links), "Links should update after step"
    
    # Unitarity should be preserved
    U = engine.links
    U_dag = U.conj().transpose(-2, -1)
    prod = torch.matmul(U, U_dag)
    identity = torch.eye(3, dtype=torch.complex64).view(1, 1, 1, 1, 3, 3).expand_as(prod)
    
    assert torch.allclose(prod, identity, atol=1e-4), "Unitarity must be preserved after update"

if __name__ == "__main__":
    test_lattice_initialization()
    test_wilson_action()
    test_step_update()
    print("All LatticeEngine tests passed!")
