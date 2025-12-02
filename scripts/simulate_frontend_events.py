import sys
import os
import torch
import json
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.physics.steering import QuantumSteering
from src.physics.quantum_kernel import QuantumMicroscope
from src.engines.qca_engine import CartesianEngine, QuantumState
from unittest.mock import MagicMock

def simulate_events():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Simulating Frontend Events on {device}...")
    
    events_log = []

    # --- 1. SETUP ENGINE ---
    grid_size = 32
    d_state = 4
    mock_model = MagicMock()
    mock_model.return_value = (torch.zeros(1, 8, grid_size, grid_size), None, None)
    mock_model.has_memory = False
    
    engine = CartesianEngine(mock_model, grid_size, d_state, device)
    # Initialize with some noise
    engine.state.psi = torch.randn(1, 8, grid_size, grid_size, device=device) * 0.1
    
    # --- 2. SIMULATE STEERING (Quantum Brush) ---
    print("🖌️ Simulating Quantum Steering...")
    steering = QuantumSteering(device)
    
    # Action: User paints a Vortex
    # Frontend sends: { "action": "quantum_steer", "pattern": "vortex", "x": 0.5, "y": 0.5, "radius": 0.2 }
    # Server does:
    psi_before = engine.state.psi.clone()
    psi_after = steering.inject(engine.state.psi, 'vortex', 16, 16, radius=6)
    
    # In a real scenario, the server broadcasts the NEW STATE (or difference) via 'state_update'
    # But specifically for the interaction, it might send an ack or effect visualization.
    # Let's capture the state update payload.
    
    # Simplified state representation for frontend (e.g., magnitude of channel 0)
    viz_data = psi_after[0, 0].abs().cpu().numpy().tolist()
    
    event_steering = {
        "timestamp": datetime.now().isoformat(),
        "type": "interaction_ack",
        "action": "quantum_steer",
        "details": {
            "pattern": "vortex",
            "applied_at": [16, 16],
            "state_change_magnitude": (psi_after - psi_before).abs().sum().item()
        }
    }
    events_log.append(event_steering)
    
    # --- 3. SIMULATE MICROSCOPE (Quantum Vision) ---
    print("🔬 Simulating Quantum Microscope...")
    microscope = QuantumMicroscope(device)
    
    # Action: User clicks to analyze a patch
    # Frontend sends: { "action": "quantum_analyze", "x": 0.5, "y": 0.5 }
    # Server extracts patch and analyzes
    
    # Create a structured patch to get interesting results
    y, x = torch.meshgrid(torch.linspace(0, 1, 4), torch.linspace(0, 1, 4), indexing='ij')
    patch = (x + y) / 2.0
    patch = patch.unsqueeze(0).to(device) # (1, 4, 4)
    
    metrics = microscope.analyze_patch(patch)
    
    # Server sends back 'quantum_analysis_result'
    event_analysis = {
        "timestamp": datetime.now().isoformat(),
        "type": "quantum_analysis_result",
        "data": {
            "complexity": metrics['complexity'],
            "activity": metrics['activity'],
            "coherence": metrics['coherence'],
            "coordinates": {"x": 0.5, "y": 0.5}
        }
    }
    events_log.append(event_analysis)
    
    # --- SAVE TO FILE ---
    output_path = "docs/40_Experiments/frontend_repro_data.json"
    with open(output_path, "w") as f:
        json.dump(events_log, f, indent=2)
        
    print(f"✅ Simulation complete. Events saved to {output_path}")
    print(json.dumps(events_log, indent=2))

if __name__ == "__main__":
    simulate_events()
