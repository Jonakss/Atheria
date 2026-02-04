import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

class FractalCell:
    def __init__(self, x, y, size, depth=0):
        self.x = x
        self.y = y
        self.size = size
        self.depth = depth
        self.state = random.random() # Simple scalar state
        self.child_grid = None
        self.is_complex = False

        # Physics constants
        self.entropy_threshold = 0.1  # Threshold to unfold dimensions (Lowered for demo)
        self.stability_threshold = 0.05 # Threshold to collapse dimensions
        self.decay = 0.01

    def evolve(self, neighbors_states):
        """
        Evolve the cell.
        If simple: Update state based on neighbors. Check entropy.
        If complex: Evolve child grid. Check stability.
        """
        if not self.is_complex:
            # Simple evolution (Average of neighbors + noise)
            if neighbors_states:
                avg = sum(neighbors_states) / len(neighbors_states)
                self.state = avg + (random.random() - 0.5) * 0.1
                self.state = max(0.0, min(1.0, self.state))

            # Calculate local "entropy" or "friction"
            # High variance from neighbors = High entropy
            variance = 0
            if neighbors_states:
                avg = sum(neighbors_states) / len(neighbors_states)
                variance = sum((s - avg) ** 2 for s in neighbors_states) / len(neighbors_states)

            # Dimensional Unfolding condition
            if variance > self.entropy_threshold and self.depth < 2: # Limit depth
                self.unfold()

        else:
            # Complex evolution
            self.child_grid.evolve()

            # Check for renormalization/stability
            child_entropy = self.child_grid.get_total_entropy()
            if child_entropy < self.stability_threshold:
                self.normalize()

    def unfold(self):
        """Create a sub-grid (Dimensional Unfolding)"""
        print(f"DEBUG: Unfolding cell at depth {self.depth} -> {self.depth + 1}")
        self.is_complex = True
        # Initialize child grid with current state as seed
        self.child_grid = FractalGrid(
            x=self.x,
            y=self.y,
            size=self.size,
            resolution=4, # 4x4 subgrid
            depth=self.depth + 1,
            seed_state=self.state
        )

    def normalize(self):
        """Collapse back to simple state (Renormalization)"""
        print(f"DEBUG: Normalizing cell at depth {self.depth}")
        self.state = self.child_grid.get_average_state()
        self.is_complex = False
        self.child_grid = None

class FractalGrid:
    def __init__(self, x, y, size, resolution, depth=0, seed_state=None):
        self.x = x
        self.y = y
        self.size = size
        self.resolution = resolution
        self.depth = depth
        self.cell_size = size / resolution

        self.cells = []
        for i in range(resolution):
            row = []
            for j in range(resolution):
                cx = x + j * self.cell_size
                cy = y + i * self.cell_size # Top-down y usually
                cell = FractalCell(cx, cy, self.cell_size, depth)
                if seed_state is not None:
                    cell.state = seed_state + (random.random() - 0.5) * 0.2
                    cell.state = max(0.0, min(1.0, cell.state))
                row.append(cell)
            self.cells.append(row)

    def get_neighbors(self, r, c):
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0: continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.resolution and 0 <= nc < self.resolution:
                    # For simple neighbor check, we just take the 'state'
                    # If neighbor is complex, we take its average state
                    neighbor = self.cells[nr][nc]
                    if neighbor.is_complex:
                        neighbors.append(neighbor.child_grid.get_average_state())
                    else:
                        neighbors.append(neighbor.state)
        return neighbors

    def evolve(self):
        # 1. Gather states first (synchronous update logic roughly)
        # For simplicity in PoC, we update in place or allow minor async artifacts

        for r in range(self.resolution):
            for c in range(self.resolution):
                neighbors = self.get_neighbors(r, c)
                self.cells[r][c].evolve(neighbors)

    def get_total_entropy(self):
        """Calculate average variance/entropy of the grid"""
        states = []
        for row in self.cells:
            for cell in row:
                if cell.is_complex:
                    states.append(cell.child_grid.get_average_state())
                else:
                    states.append(cell.state)

        avg = sum(states) / len(states)
        variance = sum((s - avg) ** 2 for s in states) / len(states)
        return variance

    def get_average_state(self):
        states = []
        for row in self.cells:
            for cell in row:
                if cell.is_complex:
                    states.append(cell.child_grid.get_average_state())
                else:
                    states.append(cell.state)
        return sum(states) / len(states)

    def draw(self, ax):
        for row in self.cells:
            for cell in row:
                if cell.is_complex:
                    cell.child_grid.draw(ax)
                    # Optional: Draw border for complex cell
                    rect = patches.Rectangle(
                        (cell.x, cell.y), cell.size, cell.size,
                        linewidth=0.5, edgecolor='red', facecolor='none', zorder=10
                    )
                    ax.add_patch(rect)
                else:
                    # Draw simple cell
                    rect = patches.Rectangle(
                        (cell.x, cell.y), cell.size, cell.size,
                        linewidth=0, facecolor=plt.cm.viridis(cell.state)
                    )
                    ax.add_patch(rect)

def main():
    print("Initializing Hyper-Fractal PoC...")

    # Root Grid: 8x8
    root_size = 100.0
    root_res = 8
    universe = FractalGrid(0, 0, root_size, root_res, depth=0)

    # Inject high entropy in center to trigger unfolding
    center = root_res // 2
    universe.cells[center][center].state = 1.0
    universe.cells[center][center+1].state = 0.0
    universe.cells[center+1][center].state = 0.0
    universe.cells[center+1][center+1].state = 1.0

    # Force unfold one cell to demonstrate recursive structure
    print("Forcing unfold of central cell...")
    universe.cells[center][center].unfold()

    print("Running simulation...")
    # Run for some steps
    for i in range(20):
        universe.evolve()
        print(f"Step {i}: Total Entropy = {universe.get_total_entropy():.4f}")

    print("Generating visualization...")
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, root_size)
    ax.set_ylim(0, root_size) # Matplotlib origin is usually bottom-left
    # Invert Y to match matrix coords if needed, but for abstract fractal it's fine.
    # Let's keep it standard Cartesian.

    ax.set_aspect('equal')
    ax.axis('off')

    universe.draw(ax)

    output_file = "hyper_fractal_poc.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_file}")

if __name__ == "__main__":
    main()
