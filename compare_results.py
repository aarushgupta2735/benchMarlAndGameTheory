"""
Compare training results between original and active-contact balance scenarios.

Reads CSV logs produced by BenchMARL and generates side-by-side plots.

Usage:
    conda activate benchmarl
    python compare_results.py
    python compare_results.py --results-dir ./results --output-dir ./plots
"""

import argparse
import os
import json
import glob
from pathlib import Path
from collections import defaultdict

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for server compatibility
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not found. Install it for plots: pip install matplotlib")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("WARNING: pandas not found. Install it for CSV parsing: pip install pandas")


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "plots"


def find_csv_logs(results_dir: Path, scenario_name: str):
    """Find all CSV log files for a given scenario across seeds."""
    pattern = str(results_dir / scenario_name / "seed_*" / "**" / "*.csv")
    files = glob.glob(pattern, recursive=True)
    # Also check for progress.csv (common TorchRL/BenchMARL output name)
    if not files:
        pattern = str(results_dir / scenario_name / "seed_*" / "**" / "progress.csv")
        files = glob.glob(pattern, recursive=True)
    return sorted(files)


def find_json_logs(results_dir: Path, scenario_name: str):
    """Find all JSON log files (marl-eval format) for a given scenario."""
    pattern = str(results_dir / scenario_name / "seed_*" / "**" / "*.json")
    return sorted(glob.glob(pattern, recursive=True))


def load_csv_data(csv_files):
    """Load and aggregate data from CSV files across seeds.

    Returns dict of {column_name: {seed: (steps, values)}}
    """
    if not HAS_PANDAS:
        print("Cannot load CSV without pandas. Install: pip install pandas")
        return {}

    all_data = defaultdict(dict)
    for f in csv_files:
        # Extract seed from path: .../seed_N/...
        parts = Path(f).parts
        seed = None
        for p in parts:
            if p.startswith("seed_"):
                seed = int(p.split("_")[1])
                break
        if seed is None:
            continue

        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"  Warning: Could not read {f}: {e}")
            continue

        # Look for reward-related columns
        for col in df.columns:
            col_lower = col.lower()
            if any(k in col_lower for k in ["reward", "return", "rew"]):
                # Try to find a step/frame column
                step_col = None
                for sc in ["step", "frame", "total_frames", "n_frames", "timestep"]:
                    if sc in [c.lower() for c in df.columns]:
                        idx = [c.lower() for c in df.columns].index(sc)
                        step_col = df.columns[idx]
                        break

                if step_col:
                    steps = df[step_col].values
                else:
                    steps = np.arange(len(df))

                all_data[col][seed] = (steps, df[col].values)

    return all_data


def compute_mean_std(data_dict):
    """Given {seed: (steps, values)}, compute mean and std across seeds.

    Interpolates to a common step grid.
    """
    if not data_dict:
        return None, None, None

    # Find common step range
    all_steps = []
    for seed, (steps, vals) in data_dict.items():
        all_steps.append(steps)

    min_max_step = min(s[-1] for s in all_steps)
    common_steps = np.linspace(0, min_max_step, 200)

    interpolated = []
    for seed, (steps, vals) in data_dict.items():
        interp_vals = np.interp(common_steps, steps, vals)
        interpolated.append(interp_vals)

    interpolated = np.array(interpolated)
    mean = interpolated.mean(axis=0)
    std = interpolated.std(axis=0)
    return common_steps, mean, std


def plot_comparison(results_dir: Path, output_dir: Path):
    """Generate comparison plots between original and active scenarios."""
    if not HAS_MPL:
        print("Cannot generate plots without matplotlib.")
        return

    os.makedirs(output_dir, exist_ok=True)

    scenarios = {
        "original": {"color": "#2196F3", "label": "Original Balance"},
        "active": {"color": "#FF5722", "label": "Balance + Contact Reward"},
    }

    # Load data for each scenario
    scenario_data = {}
    for name in scenarios:
        csv_files = find_csv_logs(results_dir, name)
        if csv_files:
            print(f"  Found {len(csv_files)} CSV files for '{name}'")
            scenario_data[name] = load_csv_data(csv_files)
        else:
            print(f"  No CSV logs found for '{name}' in {results_dir / name}")

    if not scenario_data:
        print("\nNo data found. Make sure you've run training first:")
        print("  python train_mappo.py --run-all")
        return

    # Find common reward columns
    all_columns = set()
    for name, data in scenario_data.items():
        all_columns.update(data.keys())

    reward_columns = [c for c in all_columns if any(
        k in c.lower() for k in ["reward", "return"]
    )]

    if not reward_columns:
        print("No reward/return columns found in CSV data.")
        print(f"Available columns: {all_columns}")
        return

    # Plot each reward metric
    for col in reward_columns:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        for name, meta in scenarios.items():
            if name not in scenario_data or col not in scenario_data[name]:
                continue

            steps, mean, std = compute_mean_std(scenario_data[name][col])
            if steps is None:
                continue

            ax.plot(steps, mean, color=meta["color"], label=meta["label"], linewidth=2)
            ax.fill_between(steps, mean - std, mean + std, color=meta["color"], alpha=0.2)

        ax.set_xlabel("Training Frames", fontsize=12)
        ax.set_ylabel(col, fontsize=12)
        ax.set_title(f"MAPPO: {col}\nOriginal vs Contact-Reward Balance", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        safe_col = col.replace("/", "_").replace(" ", "_")
        out_path = output_dir / f"comparison_{safe_col}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")

    # ── Normalized (Convergence) Comparison ──
    try:
        if reward_columns and "original" in scenario_data and "active" in scenario_data:
            n_col = reward_columns[0]
            if n_col in scenario_data["original"] and n_col in scenario_data["active"]:
                s_o, m_o, std_o = compute_mean_std(scenario_data["original"][n_col])
                s_a, m_a, std_a = compute_mean_std(scenario_data["active"][n_col])

                if m_o is not None and m_a is not None:
                    # Shift Active down by (FinalActive - FinalOriginal)
                    offset = m_a[-1] - m_o[-1]
                    m_a_norm = m_a - offset

                    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                    ax.plot(s_o, m_o, label="Original", color="#2196F3", linewidth=2)
                    ax.fill_between(s_o, m_o-std_o, m_o+std_o, color="#2196F3", alpha=0.2)
                    
                    lbl = f"Active (Normalized by -{offset:.1f})"
                    ax.plot(s_a, m_a_norm, label=lbl, color="#FF5722", linewidth=2, linestyle="--")
                    ax.fill_between(s_a, m_a_norm-std_a, m_a_norm+std_a, color="#FF5722", alpha=0.1)

                    ax.set_title(f"Normalized Learning Curve: {n_col}\n(Aligned at Final Mean)", fontsize=14)
                    ax.set_xlabel("Training Frames", fontsize=12)
                    ax.set_ylabel(f"{n_col} (Shifted)", fontsize=12)
                    ax.legend(fontsize=11)
                    ax.grid(True, alpha=0.3)
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)

                    out_norm = output_dir / "comparison_normalized_reward.png"
                    fig.savefig(out_norm, dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    print(f"  Saved: {out_norm}")
    except Exception as e:
        print(f"Warning: Could not create normalized plot: {e}")

    # ── Summary table ──
    print(f"\n{'='*60}")
    print(f"  Summary (final mean ± std across seeds)")
    print(f"{'='*60}")
    print(f"  {'Metric':<30} {'Original':<20} {'Active (Contact)':<20}")
    print(f"  {'-'*70}")
    for col in reward_columns:
        orig_str = "N/A"
        act_str = "N/A"
        for name in ["original", "active"]:
            if name in scenario_data and col in scenario_data[name]:
                steps, mean, std = compute_mean_std(scenario_data[name][col])
                if mean is not None and len(mean) > 0:
                    final_mean = mean[-1]
                    final_std = std[-1]
                    if name == "original":
                        orig_str = f"{final_mean:.2f} ± {final_std:.2f}"
                    else:
                        act_str = f"{final_mean:.2f} ± {final_std:.2f}"
        
        print(f"  {col:<30} {orig_str:<20} {act_str:<20}")

    print(f"\nPlots saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Compare balance scenario results")
    parser.add_argument("--results-dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    print(f"Results directory: {results_dir}")
    print(f"Output directory:  {output_dir}\n")

    plot_comparison(results_dir, output_dir)


if __name__ == "__main__":
    main()
