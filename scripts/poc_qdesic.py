import numpy as np
import matplotlib.pyplot as plt
import os

def classical_velocity_sq(r, G=1, M=1, Lambda=0):
    """
    Classical orbital velocity squared (v^2) based on Eq 76.
    v^2 = GM/r - r^2*Lambda/3
    """
    return G * M / r - (r**2 * Lambda) / 3

def qdesic_velocity_sq(r, G=1, M=1, Lambda=0, eps_params=None):
    """
    Q-desic orbital velocity squared (v^2) based on Eq 81.

    eps_params: dict containing epsilon parameters:
        e00, e10, e02, e12, e22
    """
    if eps_params is None:
        eps_params = {'e00':0, 'e10':0, 'e02':0, 'e12':0, 'e22':0}

    e00 = eps_params.get('e00', 0)
    e10 = eps_params.get('e10', 0)
    e02 = eps_params.get('e02', 0)
    e12 = eps_params.get('e12', 0)
    e22 = eps_params.get('e22', 0)

    # Numerator construction (Eq 81)
    # Term 1: -18 G^2 M^2 (1 + e22)
    term1 = -18 * (G**2) * (M**2) * (1 + e22)

    # Term 2: -r^4 (1 + e02) Lambda (3 - r^2 Lambda)
    term2 = -(r**4) * (1 + e02) * Lambda * (3 - (r**2) * Lambda)

    # Term 3: +3 G M r (1 + e12) (3 + r^2 Lambda)
    term3 = 3 * G * M * r * (1 + e12) * (3 + (r**2) * Lambda)

    numerator = term1 + term2 + term3

    # Denominator construction
    # Bracket 1: [6 GM (1 + e10) - r (1 + e00)(3 - r^2 Lambda)]
    brack1 = 6 * G * M * (1 + e10) - r * (1 + e00) * (3 - (r**2) * Lambda)

    # Bracket 2: [6 GM (1 + e12) - r (1 + e02)(3 - r^2 Lambda)]
    brack2 = 6 * G * M * (1 + e12) - r * (1 + e02) * (3 - (r**2) * Lambda)

    denominator = brack1 * brack2

    # Calculate v_phi^2 (angular velocity squared) from Eq 79-like structure in 81
    # Eq 81 gives v^2 directly? The paper says "The orbital velocity (81)..."
    # Let's check the formula structure.
    # Eq 80 relates v^2 to u_phi^2.
    # The big fraction in Eq 81 is likely v^2 directly.

    # Avoid division by zero
    if np.any(denominator == 0):
         return np.full_like(r, np.nan)

    v_sq = numerator / denominator
    return v_sq

def run_simulation():
    # Parameters
    G = 1.0
    M = 1.0
    Lambda = -0.001 # Small negative cosmological constant (AdS-like)

    # Radius range (avoid r=0)
    r = np.linspace(2.0, 50.0, 500)

    # 1. Classical Case
    v2_class = classical_velocity_sq(r, G, M, Lambda)
    v_class = np.sqrt(np.maximum(v2_class, 0))

    # 2. Q-desic Case A (Small corrections)
    eps_A = {'e00': -0.1, 'e10': 0.1, 'e02': 0.0, 'e12': 0.05, 'e22': 0.0}
    v2_qA = qdesic_velocity_sq(r, G, M, Lambda, eps_A)
    v_qA = np.sqrt(np.maximum(v2_qA, 0))

    # 3. Q-desic Case B (Strong corrections - "Dark Matter" like)
    # Testing parameters that might flatten the curve at large r
    eps_B = {'e00': -0.4, 'e10': 0.2, 'e02': 0.1, 'e12': 0.3, 'e22': 0.1}
    v2_qB = qdesic_velocity_sq(r, G, M, Lambda, eps_B)
    v_qB = np.sqrt(np.maximum(v2_qB, 0))

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.style.use('dark_background')

    plt.plot(r, v_class, 'w--', label='Classical Geodesic (GR)', linewidth=2)
    plt.plot(r, v_qA, 'c-', label='Q-desic (Weak Correction)', alpha=0.8)
    plt.plot(r, v_qB, 'm-', label='Q-desic (Strong/Dark Matter)', linewidth=2)

    plt.title(f"Orbital Velocity: Classical vs Q-desic\n(M={M}, Lambda={Lambda})")
    plt.xlabel("Radius (r)")
    plt.ylabel("Orbital Velocity (v)")
    plt.legend()
    plt.grid(True, alpha=0.2)

    output_dir = "docs/40_Experiments/plots"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/qdesic_velocity_curve.png"
    plt.savefig(output_path)
    print(f"Simulation complete. Plot saved to {output_path}")

if __name__ == "__main__":
    run_simulation()
