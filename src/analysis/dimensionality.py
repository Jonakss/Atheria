import numpy as np
import threading
import time
import logging
from collections import deque
import torch

try:
    import umap
except ImportError:
    umap = None
    logging.warning("UMAP not installed. Analysis module will be disabled.")

try:
    from sklearn.preprocessing import StandardScaler
except ImportError:
    StandardScaler = None

class StateAnalyzer:
    """
    Analyzes the high-dimensional state space using UMAP.
    Running in a separate thread to avoid blocking the main loop.
    """
    def __init__(self, buffer_size=1000, update_interval=1.0, n_neighbors=15, min_dist=0.1):
        self.buffer_size = buffer_size
        self.update_interval = update_interval
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        
        self.history_buffer = deque(maxlen=buffer_size)
        self.timestamps_buffer = deque(maxlen=buffer_size)
        
        self.umap_model = None
        self.scaler = None
        
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._new_data_event = threading.Event()
        
        self.latest_projection = []
        self.is_ready = False
        
        # Incremental learning support
        self.pending_states = []
        
    def start(self):
        if umap is None or StandardScaler is None:
            logging.error("Missing dependencies for UMAP analysis.")
            return

        self._running = True
        self._thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self._thread.start()
        logging.info("StateAnalyzer thread started.")

    def stop(self):
        self._running = False
        if self._thread:
            self._new_data_event.set() # Wake up thread
            self._thread.join(timeout=2.0)
        logging.info("StateAnalyzer thread stopped.")

    def add_state(self, state_tensor, step=None):
        """
        Adds a state to the buffer for analysis.
        Args:
            state_tensor: torch.Tensor [B, H, W, C]
            step: Simulation step number
        """
        if not self._running:
            return
            
        with self._lock:
            # Flatten state: [H, W, C] -> [H*W*C]
            # Handle Complex -> Real (Magnitude) for UMAP support
            if state_tensor.is_complex():
                state_tensor = state_tensor.abs()
                
            # Use only first batch item
            if state_tensor.dim() == 4:
                flat_state = state_tensor[0].detach().cpu().numpy().flatten()
            else:
                flat_state = state_tensor.detach().cpu().numpy().flatten()
                
            # Check consistency with previous data (history or pending)
            last_shape = None
            if self.pending_states:
                last_shape = self.pending_states[-1]['data'].shape
            elif self.history_buffer:
                last_shape = self.history_buffer[-1].shape
            
            if last_shape is not None and last_shape != flat_state.shape:
                logging.warning(f"State dimension changed. Clearing UMAP buffer. Old: {last_shape}, New: {flat_state.shape}")
                self.history_buffer.clear()
                self.timestamps_buffer.clear()
                self.pending_states.clear()
                
            self.pending_states.append({
                'data': flat_state,
                'step': step or time.time(),
                'timestamp': time.time()
            })
            
            # Notify thread if enough data
            if len(self.pending_states) >= 10: # Batch process
                self._new_data_event.set()

    def _analysis_loop(self):
        while self._running:
            # Wait for new data or timeout
            self._new_data_event.wait(timeout=self.update_interval)
            self._new_data_event.clear()
            
            if not self._running:
                break
                
            with self._lock:
                if not self.pending_states:
                    continue
                
                # Extract pending data
                new_items = list(self.pending_states)
                self.pending_states.clear()
            
            # Add to history
            for item in new_items:
                self.history_buffer.append(item['data'])
                self.timestamps_buffer.append(item['step'])
                
            # Need minimum samples to fit UMAP
            if len(self.history_buffer) < 50:
                continue
                
            # Perform UMAP
            try:
                data = np.array(self.history_buffer)
                
                # Normalize?
                # Ideally we fit scaler on first batch
                
                # Fit or Transform
                # Since UMAP parametric is complex, we might just re-fit on buffer sliding window
                # for "live" visualization. This is expensive but gives best "current view".
                # For valid trajectory, we should fix the manifold.
                # Let's try re-fitting for now as it's simpler to implement.
                
                reducer = umap.UMAP(
                    n_neighbors=self.n_neighbors,
                    min_dist=self.min_dist,
                    n_components=2,
                    metric='euclidean',
                    random_state=42,
                    n_jobs=1 # Fix UserWarning: n_jobs value 1 overridden to 1 by setting random_state.
                )
                
                # Subsample if too large for real-time
                if len(data) > 300:
                    indices = np.random.choice(len(data), 300, replace=False)
                    fit_data = data[indices]
                    steps = np.array(self.timestamps_buffer)[indices]
                else:
                    fit_data = data
                    steps = list(self.timestamps_buffer)
                
                embedding = reducer.fit_transform(fit_data)
                
                # Prepare result
                projection = []
                for i, point in enumerate(embedding):
                    projection.append({
                        'x': float(point[0]),
                        'y': float(point[1]),
                        'step': int(steps[i]) if isinstance(steps[i], (int, float)) else 0
                    })
                
                self.latest_projection = projection
                self.is_ready = True
                
                # logging.debug(f"UMAP updated with {len(fit_data)} points.")
                
            except Exception as e:
                logging.error(f"Error in UMAP loop: {e}")

    def get_latest_data(self):
        if not self.is_ready:
            return None
        return self.latest_projection
