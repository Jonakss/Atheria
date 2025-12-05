import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def classical_velocity_sq(r, G=1, M=1, Lambda=0):
    """
    Newtonian orbital velocity squared (v^2) based on Eq 76.
    v^2 = GM/r - r^2*Lambda/3
    """
    return G * M / r - (r**2 * Lambda) / 3

def gr_velocity_sq(r, G=1, M=1, Lambda=0):
    """
    General Relativistic orbital velocity squared (v^2).
    Corresponds to Q-desic Eq 81 with all epsilon parameters set to 0.
    """
    # Ensure r is an array for consistent handling
    r = np.atleast_1d(r)

    # Numerator (eps=0)
    # -18 G^2 M^2 - r^4 Lambda (3 - r^2 Lambda) + 3 G M r (3 + r^2 Lambda)
    term1 = -18 * (G**2) * (M**2)
    term2 = -(r**4) * Lambda * (3 - (r**2) * Lambda)
    term3 = 3 * G * M * r * (3 + (r**2) * Lambda)
    numerator = term1 + term2 + term3

    # Denominator (eps=0)
    # [6 GM - r(3 - r^2 Lambda)]^2
    brack = 6 * G * M - r * (3 - (r**2) * Lambda)
    denominator = brack**2

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        v_sq = numerator / denominator
        v_sq[denominator == 0] = np.nan

        if v_sq.size == 1:
            return v_sq[0]
        return v_sq

def qdesic_velocity_sq(r, G=1, M=1, Lambda=0, eps_params=None):
    """
    Q-desic orbital velocity squared (v^2) based on Eq 81.

    eps_params: dict containing epsilon parameters:
        e00, e10, e02, e12, e22
    """
    # Ensure r is an array for consistent handling
    r = np.atleast_1d(r)

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

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        v_sq = numerator / denominator
        # Filter singularities
        v_sq[denominator == 0] = np.nan

        if v_sq.size == 1:
            return v_sq[0]
        return v_sq

def run_simulation(G=1.0, M=1.0, Lambda=-0.001, r_start=2.0, r_end=50.0, points=500, output_dir="docs/40_Experiments/plots", show_plot=False):
    """
    Runs the simulation and generates the plot.
    """

    # Radius range
    r = np.linspace(r_start, r_end, points)

    # 1. Newtonian Case
    v2_newt = classical_velocity_sq(r, G, M, Lambda)
    v_newt = np.sqrt(np.maximum(v2_newt, 0))

    # 2. GR Case (Q-desic eps=0)
    v2_gr = gr_velocity_sq(r, G, M, Lambda)
    v_gr = np.sqrt(np.maximum(v2_gr, 0))

    # 3. Q-desic Case A (Small corrections)
    eps_A = {'e00': -0.1, 'e10': 0.1, 'e02': 0.0, 'e12': 0.05, 'e22': 0.0}
    v2_qA = qdesic_velocity_sq(r, G, M, Lambda, eps_A)
    v_qA = np.sqrt(np.maximum(v2_qA, 0))

    # 4. Q-desic Case B (Strong corrections - "Dark Matter" like)
    eps_B = {'e00': -0.4, 'e10': 0.2, 'e02': 0.1, 'e12': 0.3, 'e22': 0.1}
    v2_qB = qdesic_velocity_sq(r, G, M, Lambda, eps_B)
    v_qB = np.sqrt(np.maximum(v2_qB, 0))

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.style.use('dark_background')

    # Plot Newtonian as reference dashed
    plt.plot(r, v_newt, 'g:', label='Newtonian (Eq 76)', alpha=0.6)

    # Plot GR as standard reference
    plt.plot(r, v_gr, 'w--', label='GR (Q-desic eps=0)', linewidth=2)

    plt.plot(r, v_qA, 'c-', label='Q-desic (Weak Correction)', alpha=0.8)
    plt.plot(r, v_qB, 'm-', label='Q-desic (Strong/Dark Matter)', linewidth=2)

    plt.title(f"Orbital Velocity: GR vs Q-desic\n(M={M}, Lambda={Lambda})")
    plt.xlabel("Radius (r)")
    plt.ylabel("Orbital Velocity (v)")
    plt.legend()
    plt.grid(True, alpha=0.2)

    # Limit y-axis if singularities caused huge spikes
    # Filter only real values for limit calc
    valid_mask = ~np.isnan(v_qB) & ~np.isinf(v_qB)
    if np.any(valid_mask):
        safe_max = np.nanmax(v_qB[valid_mask & (r > 3*G*M)]) * 1.5
        if not np.isnan(safe_max) and safe_max > 0:
            plt.ylim(0, safe_max)

    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/qdesic_velocity_curve.png"
    plt.savefig(output_path)
    print(f"Simulation complete. Plot saved to {output_path}")

    if show_plot:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate and plot Q-desic vs Classical orbital velocities.")
    parser.add_argument("--G", type=float, default=1.0, help="Gravitational constant")
    parser.add_argument("--M", type=float, default=1.0, help="Mass of the central object")
    parser.add_argument("--Lambda", type=float, default=-0.001, help="Cosmological constant")
    parser.add_argument("--r_start", type=float, default=2.0, help="Start radius")
    parser.add_argument("--r_end", type=float, default=50.0, help="End radius")
    parser.add_argument("--points", type=int, default=500, help="Number of points in radius linspace")
    parser.add_argument("--output_dir", type=str, default="docs/40_Experiments/plots", help="Directory to save the plot")

    args = parser.parse_args()

    run_simulation(
        G=args.G,
        M=args.M,
        Lambda=args.Lambda,
        r_start=args.r_start,
        r_end=args.r_end,
        points=args.points,
        output_dir=args.output_dir
    )
