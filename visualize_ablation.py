"""
Visualization script for ablation study results

Usage:
    python visualize_ablation.py --results ablation_results/ablation_summary.json
"""

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize Ablation Study Results')
    parser.add_argument('--results', type=str, required=True,
                       help='Path to ablation_summary.json')
    parser.add_argument('--output', type=str, default='ablation_plots',
                       help='Output directory for plots')
    return parser.parse_args()


def load_results(results_file):
    """Load ablation study results from JSON"""
    with open(results_file, 'r') as f:
        data = json.load(f)
    return data


def plot_auc_comparison(results, output_dir):
    """Plot AUC comparison across all ablation configurations"""
    configs = [r['ablation_name'] for r in results['results']]
    aucs = [r['auc'] for r in results['results']]

    # Find baseline
    baseline_idx = next((i for i, name in enumerate(configs) if 'Baseline' in name), 0)
    baseline_auc = aucs[baseline_idx]

    # Calculate improvements
    improvements = [auc - baseline_auc for auc in aucs]

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Absolute AUC values
    colors = ['red' if 'Baseline' in name else 'blue' if 'Full' in name else 'gray'
              for name in configs]
    bars1 = ax1.barh(range(len(configs)), aucs, color=colors, alpha=0.7)
    ax1.set_yticks(range(len(configs)))
    ax1.set_yticklabels([name.replace('Baseline + ', '').replace('All ', '')
                         for name in configs], fontsize=9)
    ax1.set_xlabel('AUC (%)', fontsize=12)
    ax1.set_title('AUC Comparison Across Ablation Configurations', fontsize=14, fontweight='bold')
    ax1.axvline(baseline_auc, color='red', linestyle='--', linewidth=2, label='Baseline')
    ax1.legend()
    ax1.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (bar, auc) in enumerate(zip(bars1, aucs)):
        ax1.text(auc + 0.5, i, f'{auc:.2f}%', va='center', fontsize=9)

    # Plot 2: Improvement over baseline
    colors2 = ['red' if 'Baseline' in name else 'green' if imp > 0 else 'orange'
               for name, imp in zip(configs, improvements)]
    bars2 = ax2.barh(range(len(configs)), improvements, color=colors2, alpha=0.7)
    ax2.set_yticks(range(len(configs)))
    ax2.set_yticklabels([name.replace('Baseline + ', '').replace('All ', '')
                         for name in configs], fontsize=9)
    ax2.set_xlabel('AUC Improvement (%)', fontsize=12)
    ax2.set_title('AUC Improvement Over Baseline', fontsize=14, fontweight='bold')
    ax2.axvline(0, color='red', linestyle='--', linewidth=2)
    ax2.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (bar, imp) in enumerate(zip(bars2, improvements)):
        ax2.text(imp + 0.2, i, f'{imp:+.2f}%', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'auc_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {Path(output_dir) / 'auc_comparison.png'}")
    plt.close()


def plot_component_contribution(results, output_dir):
    """Plot individual component contributions"""
    # Extract single-component ablations
    single_components = []
    for r in results['results']:
        name = r['ablation_name']
        if name.startswith('Baseline +') and '+' not in name[10:]:
            component = name.replace('Baseline + ', '')
            single_components.append({
                'name': component,
                'auc': r['auc'],
                'time': r['training_time']
            })

    if not single_components:
        print("No single-component ablations found")
        return

    # Find baseline
    baseline = next((r for r in results['results'] if 'Baseline (Original' in r['ablation_name']), None)
    baseline_auc = baseline['auc'] if baseline else 0

    # Calculate contributions
    components = [c['name'] for c in single_components]
    contributions = [c['auc'] - baseline_auc for c in single_components]
    times = [c['time'] for c in single_components]

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Component contributions
    colors = plt.cm.viridis(np.linspace(0, 1, len(components)))
    bars = ax1.bar(range(len(components)), contributions, color=colors, alpha=0.8)
    ax1.set_xticks(range(len(components)))
    ax1.set_xticklabels(components, rotation=45, ha='right', fontsize=10)
    ax1.set_ylabel('AUC Improvement (%)', fontsize=12)
    ax1.set_title('Individual Component Contributions', fontsize=14, fontweight='bold')
    ax1.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (bar, contrib) in enumerate(zip(bars, contributions)):
        ax1.text(i, contrib + 0.1, f'{contrib:+.2f}%', ha='center', fontsize=9)

    # Plot 2: Performance vs Training Time
    scatter = ax2.scatter(times, contributions, s=200, c=range(len(components)),
                         cmap='viridis', alpha=0.7, edgecolors='black', linewidth=1.5)
    ax2.set_xlabel('Training Time (seconds)', fontsize=12)
    ax2.set_ylabel('AUC Improvement (%)', fontsize=12)
    ax2.set_title('Performance vs Training Cost', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Add labels
    for i, (x, y, name) in enumerate(zip(times, contributions, components)):
        ax2.annotate(name, (x, y), xytext=(5, 5), textcoords='offset points',
                    fontsize=8, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'component_contributions.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {Path(output_dir) / 'component_contributions.png'}")
    plt.close()


def plot_cumulative_effect(results, output_dir):
    """Plot cumulative effect of adding components"""
    # Define progression
    progression = [
        'Baseline (Original Model)',
        'Baseline + Temporal Attention',
        'Temporal Attention + Memory',
        'All Attention Modules',
        'Full Model (All Improvements)'
    ]

    aucs = []
    names = []
    for prog in progression:
        result = next((r for r in results['results'] if r['ablation_name'] == prog), None)
        if result:
            aucs.append(result['auc'])
            names.append(prog.replace('Baseline + ', '').replace('Baseline (', '('))

    if len(aucs) < 2:
        print("Not enough data for cumulative plot")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    x = range(len(names))
    line = ax.plot(x, aucs, marker='o', markersize=10, linewidth=2.5,
                   color='#2E86AB', markerfacecolor='#A23B72', markeredgewidth=2,
                   markeredgecolor='#2E86AB')

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha='right', fontsize=10)
    ax.set_ylabel('AUC (%)', fontsize=12)
    ax.set_xlabel('Model Configuration', fontsize=12)
    ax.set_title('Cumulative Effect of Improvements', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add value labels
    for i, (xi, auc) in enumerate(zip(x, aucs)):
        improvement = f'(+{auc - aucs[0]:.2f}%)' if i > 0 else '(baseline)'
        ax.annotate(f'{auc:.2f}%\n{improvement}',
                   (xi, auc), xytext=(0, 10), textcoords='offset points',
                   ha='center', fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

    # Shade the improvement area
    ax.fill_between(x, aucs[0], aucs, alpha=0.2, color='green')

    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'cumulative_effect.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {Path(output_dir) / 'cumulative_effect.png'}")
    plt.close()


def generate_summary_table(results, output_dir):
    """Generate a summary table in text format"""
    output_file = Path(output_dir) / 'ablation_summary.txt'

    # Find baseline
    baseline = next((r for r in results['results'] if 'Baseline (Original' in r['ablation_name']), None)
    baseline_auc = baseline['auc'] if baseline else 0

    with open(output_file, 'w') as f:
        f.write("="*100 + "\n")
        f.write("ABLATION STUDY SUMMARY\n")
        f.write("="*100 + "\n\n")

        f.write(f"Dataset: {results['dataset']}\n")
        f.write(f"Training Epochs: {results['epochs']}\n")
        f.write(f"Timestamp: {results['timestamp']}\n\n")

        f.write("-"*100 + "\n")
        f.write(f"{'Configuration':<45} {'AUC (%)':<12} {'Improvement':<15} {'Time (s)':<12}\n")
        f.write("-"*100 + "\n")

        for r in results['results']:
            name = r['ablation_name']
            auc = r['auc']
            improvement = f"+{auc - baseline_auc:.2f}%" if baseline_auc > 0 else "N/A"
            time = r['training_time']

            f.write(f"{name:<45} {auc:<12.2f} {improvement:<15} {time:<12.2f}\n")

        f.write("-"*100 + "\n\n")

        # Key findings
        f.write("KEY FINDINGS:\n")
        f.write("="*100 + "\n\n")

        # Best single component
        single_comps = [r for r in results['results']
                       if r['ablation_name'].startswith('Baseline +') and
                       '+' not in r['ablation_name'][10:]]
        if single_comps:
            best_single = max(single_comps, key=lambda x: x['auc'])
            f.write(f"Best Single Component: {best_single['ablation_name']}\n")
            f.write(f"  Improvement: +{best_single['auc'] - baseline_auc:.2f}%\n\n")

        # Full model improvement
        full = next((r for r in results['results'] if 'Full Model' in r['ablation_name']), None)
        if full:
            f.write(f"Full Model Improvement: +{full['auc'] - baseline_auc:.2f}%\n")
            f.write(f"  Final AUC: {full['auc']:.2f}%\n\n")

    print(f"Saved: {output_file}")


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    print(f"Loading results from: {args.results}")
    results = load_results(args.results)

    print(f"\nFound {len(results['results'])} ablation experiments")
    print(f"Generating visualizations...\n")

    # Generate plots
    plot_auc_comparison(results, output_dir)
    plot_component_contribution(results, output_dir)
    plot_cumulative_effect(results, output_dir)
    generate_summary_table(results, output_dir)

    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == '__main__':
    main()
