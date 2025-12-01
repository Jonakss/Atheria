# Quality Diversity & MAP-Elites

## 1. Beyond Optimization
Traditional optimization algorithms (Gradient Descent, Genetic Algorithms) are "convergent." They seek a single global maximum. They ask: *"What is the best solution?"*

**Quality Diversity (QD)** algorithms ask a different question: *"What are all the different **kinds** of high-quality solutions?"*

In the context of Aetheria (and creative AI in general), we don't just want one "perfect" universe. We want an atlas of all possible stable universes: some static, some chaotic, some oscillating, some with life-like gliders.

## 2. MAP-Elites Algorithm
**Multi-dimensional Archive of Phenotypic Elites (MAP-Elites)** is the gold standard algorithm for QD.

### 2.1 The Concept
Instead of a single population list, MAP-Elites maintains a **Grid (Archive)**.
The axes of this grid are not parameters (genes), but **Behavioral Descriptors (BC)**.

### 2.2 Behavioral Descriptors (BC)
These are the "features" of the resulting simulation.
*   *Example 1:* **Entropy** (How disordered is the universe?)
*   *Example 2:* **Stability** (How much does it change over time?)
*   *Example 3:* **Active Mass** (How many cells are alive?)

A specific simulation run is mapped to a cell in this grid (e.g., High Entropy, Low Stability).

### 2.3 The Algorithm Step-by-Step
1.  **Grid Initialization:** Create an empty N-dimensional grid.
2.  **Random Initialization:** Generate random genotypes (physics parameters), simulate them, measure their BCs, and place them in the grid.
3.  **Selection:** Pick a random "Elite" from a non-empty grid cell.
4.  **Variation:** Mutate the Elite's genotype (e.g., slightly change gravity or decay).
5.  **Evaluation:** Run the simulation with the new genotype.
    *   Calculate **Fitness** (e.g., Total Energy).
    *   Calculate **BC** (Entropy, Stability).
6.  **Placement:**
    *   Find the target cell for the new BC.
    *   **Competition:** If the cell is empty, place the new individual. If occupied, keep the one with the higher **Fitness**.

### 2.4 The Result: The Illumination
Over time, MAP-Elites "illuminates" the search space. It fills the grid with high-performing solutions for *every* niche.
This is crucial for **Experiment 05 (Quality Diversity)**, as it allows us to visualize the "Phase Space" of the Aetheria physics engine, identifying regions of interest (e.g., "The Edge of Chaos") without knowing exactly where they are beforehand.

## 3. Why it matters for Digital Physics
Convergent evolution often gets stuck in local optima (boring stable states). QD algorithms preserve diversity. A "failed" mutation that creates high chaos might be the stepping stone to a complex "Life" state, but a traditional algorithm would discard it for having "bad fitness" compared to a stable void. MAP-Elites keeps it because it is the "best at being chaotic," allowing it to evolve further.

## 4. Key References
*   *Mouret & Clune, "Illuminating search spaces by mapping elites", arXiv, 2015.*
*   *Pugh et al., "Quality Diversity: A New Frontier for Evolutionary Computation", Frontiers in Robotics and AI, 2016.*
*   *Cully et al., "Robots that can adapt like animals", Nature, 2015.*
