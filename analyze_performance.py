"""
Statistical analysis and performance comparison for MAPPO training results.

Extends compare_results.py with:
  - Convergence speed analysis (time-to-threshold)
  - Stability metrics (variance, coefficient of variation)
  - Statistical significance tests (Wilcoxon, t-test)
  - Advanced visualizations (box plots, stability curves)

Usage:
    python analyze_performance.py
    python analyze_performance.py --results-dir ./results --output-dir ./plots
"""

import argparse
import os
import json
import glob
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not found. Install for stats tests: pip install scipy")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not found. Install for plots: pip install matplotlib")


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "plots"


# ════════════════════════════════════════════════════════════════════════════
# Data Loading (similar to compare_results.py)
# ════════════════════════════════════════════════════════════════════════════

def find_csv_logs(results_dir: Path, scenario_name: str):
    """Find all CSV log files for a given scenario across seeds."""
    pattern = str(results_dir / scenario_name / "seed_*" / "**" / "*.csv")
    files = glob.glob(pattern, recursive=True)
    if not files:
        pattern = str(results_dir / scenario_name / "seed_*" / "**" / "progress.csv")
        files = glob.glob(pattern, recursive=True)
    return sorted(files)


def load_csv_data_per_seed(csv_files):
    """Load CSV data organized by seed and column.
    
    Returns: {seed: {column_name: (steps, values)}}
    """
    seed_data = defaultdict(lambda: defaultdict(dict))
    
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
        
        # Extract step column
        step_col = None
        for sc in ["step", "frame", "total_frames", "n_frames", "timestep"]:
            if sc in [c.lower() for c in df.columns]:
                idx = [c.lower() for c in df.columns].index(sc)
                step_col = df.columns[idx]
                break
        
        steps = df[step_col].values if step_col else np.arange(len(df))
        
        # Extract reward columns
        for col in df.columns:
            col_lower = col.lower()
            if any(k in col_lower for k in ["reward", "return", "rew"]):
                seed_data[seed][col] = (steps.copy(), df[col].values.copy())
    
    return seed_data


def aggregate_seeds_to_dict(seed_data):
    """Convert {seed: {col: (steps, vals)}} to {col: {seed: (steps, vals)}}."""
    col_data = defaultdict(dict)
    for seed, columns in seed_data.items():
        for col, (steps, vals) in columns.items():
            col_data[col][seed] = (steps, vals)
    return col_data


# ════════════════════════════════════════════════════════════════════════════
# Convergence Analysis
# ════════════════════════════════════════════════════════════════════════════

def compute_time_to_threshold(seed_data_dict, threshold_percentile=0.9):
    """Compute frames required to reach threshold for each seed.
    
    Args:
        seed_data_dict: {seed: (steps, values)}
        threshold_percentile: Multiply final mean by this to get threshold
        
    Returns:
        {seed: frames_to_reach} or {seed: np.nan} if never reached
    """
    # Compute threshold (90% of final mean reward)
    final_rewards = [vals[-1] for steps, vals in seed_data_dict.values()]
    threshold = np.mean(final_rewards) * threshold_percentile
    
    time_to_threshold_dict = {}
    for seed, (steps, vals) in seed_data_dict.items():
        # Find first index where value >= threshold
        reached_idx = np.where(vals >= threshold)[0]
        if len(reached_idx) > 0:
            time_to_threshold_dict[seed] = float(steps[reached_idx[0]])
        else:
            time_to_threshold_dict[seed] = np.nan
    
    return time_to_threshold_dict, threshold


def compute_convergence_speed_stats(time_to_threshold_dict):
    """Compute mean and std of time-to-threshold across seeds.
    
    Returns:
        (mean, std, count_successful)
    """
    valid_times = [t for t in time_to_threshold_dict.values() if not np.isnan(t)]
    
    if len(valid_times) == 0:
        return np.nan, np.nan, 0
    
    return np.mean(valid_times), np.std(valid_times), len(valid_times)


# ════════════════════════════════════════════════════════════════════════════
# Stability Analysis
# ════════════════════════════════════════════════════════════════════════════

def compute_final_reward_stats(seed_data_dict):
    """Extract final reward value for each seed.
    
    Returns:
        {seed: final_reward}
    """
    return {seed: vals[-1] for seed, (steps, vals) in seed_data_dict.items()}


def compute_stability_stats(seed_data_dict):
    """Compute stability metrics across seeds.
    
    Returns:
        {
            'final_rewards': {seed: value},
            'final_mean': float,
            'final_std': float,
            'final_cv': float,  # Coefficient of variation
            'final_sem': float  # Standard error of the mean
        }
    """
    final_rewards = compute_final_reward_stats(seed_data_dict)
    final_values = np.array(list(final_rewards.values()))
    
    mean_val = np.mean(final_values)
    std_val = np.std(final_values, ddof=1)  # Sample std
    cv = std_val / mean_val if mean_val != 0 else np.nan
    sem = std_val / np.sqrt(len(final_values))
    
    return {
        'final_rewards': final_rewards,
        'final_mean': mean_val,
        'final_std': std_val,
        'final_cv': cv,
        'final_sem': sem,
        'n_seeds': len(final_rewards)
    }


def compute_rolling_stability(seed_data_dict, window_fraction=0.1):
    """Compute rolling std over a window at the end of training.
    
    Args:
        seed_data_dict: {seed: (steps, values)}
        window_fraction: Use last N% of steps
        
    Returns:
        {seed: rolling_std_in_window}
    """
    rolling_stds = {}
    
    for seed, (steps, vals) in seed_data_dict.items():
        window_start = int(len(vals) * (1 - window_fraction))
        window_vals = vals[window_start:]
        rolling_stds[seed] = np.std(window_vals, ddof=1)
    
    return rolling_stds


# ════════════════════════════════════════════════════════════════════════════
# Statistical Significance Tests
# ════════════════════════════════════════════════════════════════════════════

def wilcoxon_significance_test(original_final_rewards, active_final_rewards):
    """Perform Wilcoxon signed-rank test on final rewards.
    
    Args:
        original_final_rewards: array-like of final rewards
        active_final_rewards: array-like of final rewards
        
    Returns:
        {
            'test_name': 'Wilcoxon Signed-Rank Test',
            'statistic': float,
            'p_value': float,
            'significant': bool (α=0.05)
        }
    """
    if not HAS_SCIPY:
        return {'error': 'scipy not available'}
    
    try:
        # Paired test: difference between scenarios
        stat, pval = stats.wilcoxon(original_final_rewards, active_final_rewards)
        return {
            'test_name': 'Wilcoxon Signed-Rank Test',
            'statistic': float(stat),
            'p_value': float(pval),
            'significant': pval < 0.05,
            'n_samples': len(original_final_rewards)
        }
    except Exception as e:
        return {'error': str(e)}


def ttest_significance_test(original_final_rewards, active_final_rewards):
    """Perform independent t-test on final rewards.
    
    Args:
        original_final_rewards: array-like
        active_final_rewards: array-like
        
    Returns:
        {
            'test_name': 't-test',
            'statistic': float,
            'p_value': float,
            'significant': bool (α=0.05)
        }
    """
    if not HAS_SCIPY:
        return {'error': 'scipy not available'}
    
    try:
        stat, pval = stats.ttest_ind(original_final_rewards, active_final_rewards)
        return {
            'test_name': "Independent t-test",
            'statistic': float(stat),
            'p_value': float(pval),
            'significant': pval < 0.05,
            'n_original': len(original_final_rewards),
            'n_active': len(active_final_rewards)
        }
    except Exception as e:
        return {'error': str(e)}


# ════════════════════════════════════════════════════════════════════════════
# Visualizations
# ════════════════════════════════════════════════════════════════════════════

def plot_time_to_threshold(original_ttt_stats, active_ttt_stats, output_dir: Path):
    """Bar plot of convergence speed (time-to-threshold)."""
    if not HAS_MPL:
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    scenarios = ['Original', 'Active (Contact)']
    means = [original_ttt_stats[0], active_ttt_stats[0]]
    stds = [original_ttt_stats[1], active_ttt_stats[1]]
    colors = ['#2196F3', '#FF5722']
    
    x_pos = np.arange(len(scenarios))
    ax.bar(x_pos, means, yerr=stds, capsize=10, color=colors, alpha=0.8, 
           edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Frames to Threshold', fontsize=12)
    ax.set_title('Convergence Speed: Time to 90% Final Reward', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(scenarios)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    out_path = output_dir / 'analysis_time_to_threshold.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_final_reward_boxplot(original_stats, active_stats, output_dir: Path):
    """Box plot of final reward distribution across seeds."""
    if not HAS_MPL:
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    orig_rewards = list(original_stats['final_rewards'].values())
    active_rewards = list(active_stats['final_rewards'].values())
    
    data = [orig_rewards, active_rewards]
    labels = ['Original', 'Active (Contact)']
    colors = ['#2196F3', '#FF5722']
    
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Overlay individual points
    for i, (label, points) in enumerate(zip(labels, data), 1):
        y = points
        x = np.random.normal(i, 0.04, size=len(y))
        ax.scatter(x, y, alpha=0.6, s=100, color='black', zorder=3)
    
    ax.set_ylabel('Final Reward', fontsize=12)
    ax.set_title('Final Reward Distribution Across Seeds', fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    out_path = output_dir / 'analysis_final_reward_boxplot.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ════════════════════════════════════════════════════════════════════════════
# Reporting
# ════════════════════════════════════════════════════════════════════════════

def generate_analysis_report(scenario_data, output_dir: Path, reward_column):
    """Generate comprehensive statistical report."""
    print(f"\n{'='*70}")
    print(f"  STATISTICAL ANALYSIS REPORT")
    print(f"  Metric: {reward_column}")
    print(f"{'='*70}\n")
    
    # Extract data for scenarios
    scenarios_to_analyze = {}
    for name in ['original', 'active']:
        if name in scenario_data and reward_column in scenario_data[name]:
            scenarios_to_analyze[name] = scenario_data[name][reward_column]
    
    if len(scenarios_to_analyze) != 2:
        print("  ERROR: Could not find both scenarios for comparison.")
        return
    
    # ── Convergence Speed ──
    print("  1. CONVERGENCE SPEED")
    print(f"  {'-'*68}")
    for name, seed_dict in scenarios_to_analyze.items():
        ttt_dict, threshold = compute_time_to_threshold(seed_dict)
        mean_ttt, std_ttt, n_success = compute_convergence_speed_stats(ttt_dict)
        
        print(f"  {name.upper()}")
        print(f"    Threshold (90% final mean): {threshold:.2f}")
        print(f"    Time to threshold: {mean_ttt:.0f} ± {std_ttt:.0f} frames " +
              f"({n_success}/{len(seed_dict)} seeds)")
        print()
    
    # ── Stability ──
    print("  2. STABILITY & VARIANCE")
    print(f"  {'-'*68}")
    stats_dict = {}
    for name, seed_dict in scenarios_to_analyze.items():
        stats_dict[name] = compute_stability_stats(seed_dict)
        s = stats_dict[name]
        print(f"  {name.upper()}")
        print(f"    Final Reward: {s['final_mean']:.4f} ± {s['final_std']:.4f}")
        print(f"    Coefficient of Variation: {s['final_cv']:.4f}")
        print(f"    Std Error of Mean: {s['final_sem']:.4f}")
        print(f"    Sample Size (N): {s['n_seeds']}")
        print()
    
    # ── Statistical Tests ──
    print("  3. STATISTICAL SIGNIFICANCE TESTS")
    print(f"  {'-'*68}")
    
    orig_final = list(stats_dict['original']['final_rewards'].values())
    active_final = list(stats_dict['active']['final_rewards'].values())
    
    # Wilcoxon
    if HAS_SCIPY:
        wilcox_result = wilcoxon_significance_test(orig_final, active_final)
        print(f"  Wilcoxon Signed-Rank Test:")
        if 'error' not in wilcox_result:
            print(f"    Statistic: {wilcox_result['statistic']:.4f}")
            print(f"    p-value: {wilcox_result['p_value']:.4f}")
            sig = "***" if wilcox_result['significant'] else "ns"
            print(f"    Significant (α=0.05): {wilcox_result['significant']} {sig}")
        else:
            print(f"    Error: {wilcox_result['error']}")
        print()
        
        # t-test
        ttest_result = ttest_significance_test(orig_final, active_final)
        print(f"  Independent t-test:")
        if 'error' not in ttest_result:
            print(f"    Statistic: {ttest_result['statistic']:.4f}")
            print(f"    p-value: {ttest_result['p_value']:.4f}")
            sig = "***" if ttest_result['significant'] else "ns"
            print(f"    Significant (α=0.05): {ttest_result['significant']} {sig}")
        else:
            print(f"    Error: {ttest_result['error']}")
        print()
    else:
        print("  (scipy not available for statistical tests)")
        print()
    
    # ── Summary ──
    print("  4. INTERPRETATION")
    print(f"  {'-'*68}")
    orig_final_mean = stats_dict['original']['final_mean']
    active_final_mean = stats_dict['active']['final_mean']
    improvement = ((active_final_mean - orig_final_mean) / abs(orig_final_mean)) * 100
    
    print(f"  Final Reward Difference: {improvement:+.2f}%")
    if abs(improvement) > 5:
        direction = "ACTIVE is better" if improvement > 0 else "ORIGINAL is better"
        print(f"  Verdict: {direction}")
    else:
        print(f"  Verdict: Statistically similar performance")
    print()
    
    print(f"{'='*70}\n")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main(results_dir: Path, output_dir: Path):
    """Run full analysis pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"  ADVANCED PERFORMANCE ANALYSIS")
    print(f"{'='*70}")
    
    # Load data
    scenario_data = {}
    for name in ['original', 'active']:
        csv_files = find_csv_logs(results_dir, name)
        if csv_files:
            print(f"\n  Loading '{name}': {len(csv_files)} CSV files")
            seed_data = load_csv_data_per_seed(csv_files)
            scenario_data[name] = aggregate_seeds_to_dict(seed_data)
        else:
            print(f"\n  WARNING: No CSV logs found for '{name}'")
    
    if not scenario_data:
        print("\n  ERROR: No data found. Run training first: python train_mappo.py --run-all")
        return
    
    # Find common reward columns
    all_columns = set()
    for name, data in scenario_data.items():
        all_columns.update(data.keys())
    reward_columns = [c for c in all_columns if any(k in c.lower() for k in ["reward", "return"])]
    
    if not reward_columns:
        print("  ERROR: No reward/return columns found in CSV data.")
        return
    
    # Analyze each reward column
    for col in reward_columns:
        generate_analysis_report(scenario_data, output_dir, col)
        
        # Generate plots for main reward metric
        if col == reward_columns[0]:
            print(f"\n  Generating plots for: {col}")
            
            # Convergence speed plot
            if 'original' in scenario_data and col in scenario_data['original']:
                orig_dict = scenario_data['original'][col]
                orig_ttt, _ = compute_time_to_threshold(orig_dict)
                orig_ttt_stats = compute_convergence_speed_stats(orig_ttt)
                
                active_dict = scenario_data['active'][col]
                active_ttt, _ = compute_time_to_threshold(active_dict)
                active_ttt_stats = compute_convergence_speed_stats(active_ttt)
                
                plot_time_to_threshold(orig_ttt_stats, active_ttt_stats, output_dir)
            
            # Final reward box plot
            if 'original' in scenario_data and col in scenario_data['original']:
                orig_stats = compute_stability_stats(scenario_data['original'][col])
                active_stats = compute_stability_stats(scenario_data['active'][col])
                plot_final_reward_boxplot(orig_stats, active_stats, output_dir)
    
    print(f"\n  Analysis complete. Results saved to: {output_dir}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Path to results directory (default: {DEFAULT_RESULTS_DIR})"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Path to output directory for plots (default: {DEFAULT_OUTPUT_DIR})"
    )
    
    args = parser.parse_args()
    main(args.results_dir, args.output_dir)