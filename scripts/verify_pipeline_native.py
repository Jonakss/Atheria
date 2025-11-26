import asyncio
import logging
import sys
import os
import json
import torch
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from src.server.server_state import g_state
from src.pipelines.handlers.inference_handlers import handle_load_experiment, handle_play, handle_pause
from src.pipelines.core.simulation_loop import simulation_loop
from src.engines.native_engine_wrapper import NativeEngineWrapper

# Mock WebSocket for notifications
class MockWebSocket:
    def __init__(self):
        self.closed = False
        self.messages = []
    
    async def send_str(self, data):
        self.messages.append(data)
    
    async def send_json(self, data):
        self.messages.append(json.dumps(data))

async def verify_pipeline():
    print("🚀 Verifying Native Engine Pipeline Integration...")
    
    # 1. Setup Mock Environment
    ws_id = "mock_client"
    mock_ws = MockWebSocket()
    g_state['websockets'] = {ws_id: mock_ws}
    
    # 2. Create Dummy Model for Testing
    print("   🔨 Creating dummy model for testing...")
    d_state = 8
    grid_size = 64
    model = torch.nn.Conv2d(2*d_state, 2*d_state, kernel_size=3, padding=1)
    example_input = torch.randn(1, 2*d_state, 20, 20) # Chunk size
    traced_model = torch.jit.trace(model, example_input)
    model_path = os.path.abspath("dummy_pipeline_model.pt")
    traced_model.save(model_path)
    
    # Mock experiment config
    from types import SimpleNamespace
    exp_cfg = SimpleNamespace(
        MODEL_PARAMS=SimpleNamespace(d_state=d_state),
        TRAINING=SimpleNamespace(batch_size=1),
        MODEL_ARCHITECTURE="Conv2D" # Added to prevent AttributeError on fallback
    )
    
    # Mocking utils in handlers
    import src.pipelines.handlers.inference_handlers as handlers
    
    # Monkey patch functions in the handlers module where they are used
    handlers.load_experiment_config = lambda name: exp_cfg
    handlers.get_latest_jit_model = lambda name, silent=False: Path(model_path)
    handlers.get_latest_checkpoint = lambda name: "dummy_checkpoint.pt"
    
    # Also patch utils just in case
    import src.utils as utils
    utils.load_experiment_config = lambda name: exp_cfg
    utils.get_latest_jit_model = lambda name, silent=False: Path(model_path)
    utils.get_latest_checkpoint = lambda name: "dummy_checkpoint.pt"
    
    try:
        # 3. Load Experiment (Native)
        print("   📦 Loading experiment with force_engine='native'...")
        await handle_load_experiment({
            'ws_id': ws_id,
            'experiment_name': 'test_experiment',
            'force_engine': 'native'
        })
        
        motor = g_state.get('motor')
        if not motor:
            print("   ❌ Failed to load motor")
            return
            
        if not g_state.get('motor_is_native'):
            print("   ❌ Motor is not native")
            return
            
        print("   ✅ Native Engine loaded successfully")
        
        # 4. Inject Seed (Manually for test)
        print("   🌱 Injecting seed...")
        # Native engine wrapper should have initialized state.
        # Let's inject a seed into the dense state and re-init native
        if hasattr(motor, 'state') and motor.state.psi is not None:
            motor.state.psi[0, grid_size//2, grid_size//2] = torch.randn(d_state, dtype=torch.complex64)
            motor._initialize_native_state_from_dense(motor.state.psi)
            print(f"   Particles: {motor.native_engine.get_matter_count()}")
        
        # 5. Start Simulation (Play)
        print("   ▶️ Starting simulation...")
        await handle_play({'ws_id': ws_id})
        
        if g_state.get('is_paused'):
            print("   ❌ Simulation failed to start (is_paused=True)")
            return
            
        # 6. Run Loop for a few seconds
        print("   🏃 Running simulation loop for 3 seconds...")
        
        # Run simulation_loop in background
        loop_task = asyncio.create_task(simulation_loop())
        
        # Wait
        await asyncio.sleep(3)
        
        # 7. Pause
        print("   ⏸️ Pausing...")
        await handle_pause({'ws_id': ws_id})
        
        # Cancel loop
        loop_task.cancel()
        try:
            await loop_task
        except asyncio.CancelledError:
            pass
            
        # 8. Verify Results
        final_step = g_state.get('simulation_step', 0)
        final_particles = motor.native_engine.get_matter_count()
        
        print(f"   📊 Final Step: {final_step}")
        print(f"   📊 Final Particles: {final_particles}")
        
        if final_step > 0 and final_particles > 0:
            print("   ✅ Pipeline verification SUCCESSFUL!")
        else:
            print("   ❌ Pipeline verification FAILED (no steps or no particles)")

    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        utils.load_experiment_config = original_load_config
        utils.get_latest_jit_model = original_get_jit
        utils.get_latest_checkpoint = original_get_checkpoint
        if os.path.exists(model_path):
            os.remove(model_path)

if __name__ == "__main__":
    asyncio.run(verify_pipeline())
