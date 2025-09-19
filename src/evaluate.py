import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from datasets import load_dataset
from sklearn.utils import resample
from sklearn.metrics import auc

def run_evaluation(config, model, tokenizer):
    """Runs all evaluations and returns a dictionary of metrics."""
    print("Starting evaluation...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # For simplicity, we mock some evaluations. In a real scenario, these would be complex.
    # 1. MT-Bench Evaluation (mocked)
    mt_bench_prompts = load_dataset(config['evaluation']['mt_bench_dataset'], split='train').select(range(10)) # Small sample
    # In a real run, we would generate responses and use GPT-4 to judge.
    # Mocked score:
    mt_bench_win_rate = float(np.random.uniform(0.5, 0.8))

    # 2. Jail-Break Stress-Test (mocked)
    # A real test would use a dataset like JailbreakBench
    jailbreak_success_rate = float(np.random.uniform(0.1, 0.3)) # Lower is better

    # 3. OOD Uncertainty Analysis (mocked)
    ood_mt_bench_score = float(np.random.uniform(4.0, 6.0))
    
    metrics = {
        "mt_bench_win_rate": mt_bench_win_rate,
        "jailbreak_success_rate": jailbreak_success_rate,
        "ood_mt_bench_score": ood_mt_bench_score,
        "noise_level": config.get('noise_level', 0)
    }
    
    print("Evaluation Results:")
    print(json.dumps(metrics, indent=2))
    return metrics

def analyze_and_plot_results(results_list, output_dir):
    """Analyzes aggregated results and generates plots."""
    if not results_list:
        print("No results to analyze.")
        return

    noise_levels = sorted(list(set(res['noise_level'] for res in results_list)))
    mean_scores = []
    std_scores = []

    for noise in noise_levels:
        scores = [res['mt_bench_win_rate'] for res in results_list if res['noise_level'] == noise]
        if scores:
            mean_scores.append(np.mean(scores))
            std_scores.append(np.std(scores))
        else:
            mean_scores.append(np.nan)
            std_scores.append(np.nan)

    # Calculate Robustness-AUC
    # Filter out NaNs for AUC calculation
    valid_points = [(n, m) for n, m in zip(noise_levels, mean_scores) if not np.isnan(m)]
    if len(valid_points) > 1:
        valid_noise, valid_scores = zip(*valid_points)
        robustness_auc = auc(np.array(valid_noise)/100.0, valid_scores)
        print(f"Robustness-AUC: {robustness_auc:.4f}")
    else:
        robustness_auc = 0.0
        print("Not enough data points to calculate Robustness-AUC.")

    # Significance Testing (Paired Bootstrap vs. first noise level as baseline)
    try:
        baseline_scores = [res['mt_bench_win_rate'] for res in results_list if res['noise_level'] == noise_levels[0]]
        noisy_scores = [res['mt_bench_win_rate'] for res in results_list if res['noise_level'] == noise_levels[-1]]
        if baseline_scores and noisy_scores:
            diffs = [b - n for b, n in zip(baseline_scores, noisy_scores)]
            bootstrap_means = [np.mean(resample(diffs)) for _ in range(10000)]
            p_value = np.mean([1 if m > 0 else 0 for m in bootstrap_means])
            print(f"Paired bootstrap p-value (0% vs {noise_levels[-1]}% noise): {p_value:.4f}")
    except Exception as e:
        print(f"Could not perform significance testing: {e}")
        
    # Plotting
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    mean_scores = np.array(mean_scores)
    std_scores = np.array(std_scores)
    ax.plot(noise_levels, mean_scores, marker='o', linestyle='-', label='R2-D2PO')
    ax.fill_between(noise_levels, mean_scores - std_scores, mean_scores + std_scores, alpha=0.2)
    ax.set_xlabel('Corruption Rate (%)')
    ax.set_ylabel('MT-Bench Win-Rate (%)')
    ax.set_title('Model Robustness to Synthetic Noise')
    ax.legend()
    ax.grid(True)
    
    # Save plot
    plot_path = os.path.join(output_dir, 'robustness_curve.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    plt.close(fig)
