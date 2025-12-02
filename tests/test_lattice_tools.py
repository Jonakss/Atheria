import torch
import logging
import inspect
import sys
import os

# Force local import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.engines.lattice_engine import LatticeEngine

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_lattice_tools():
    print(f"🔍 LatticeEngine imported from: {inspect.getfile(LatticeEngine)}")
    print("🧪 Testing Lattice Engine Tools...")
    
    # Initialize engine
    grid_size = 64
    engine = LatticeEngine(grid_size=grid_size, d_state=1)
    
    # Snapshot initial state
    initial_links = engine.links.clone()
    
    # Test 1: Collapse
    print("\n⚡ Testing Collapse...")
    engine.apply_tool('collapse', {'intensity': 0.8})
    
    diff = (engine.links - initial_links).abs().sum().item()
    print(f"Collapse Diff: {diff}")
    assert diff > 0, "Collapse should modify links"
    
    # Snapshot after collapse
    post_collapse_links = engine.links.clone()
    
    # Test 2: Vortex
    print("\n🌀 Testing Vortex...")
    engine.apply_tool('vortex', {'x': 32, 'y': 32, 'radius': 10, 'strength': 1.0})
    
    diff_vortex = (engine.links - post_collapse_links).abs().sum().item()
    print(f"Vortex Diff: {diff_vortex}")
    assert diff_vortex > 0, "Vortex should modify links"
    
    # Snapshot after vortex
    post_vortex_links = engine.links.clone()
    
    # Test 3: Wave
    print("\n🌊 Testing Wave...")
    engine.apply_tool('wave', {'k_x': 2.0, 'k_y': 2.0})
    
    diff_wave = (engine.links - post_vortex_links).abs().sum().item()
    print(f"Wave Diff: {diff_wave}")
    assert diff_wave > 0, "Wave should modify links"
    
    print("\n✅ All Lattice Tools tests passed!")

if __name__ == "__main__":
    test_lattice_tools()
