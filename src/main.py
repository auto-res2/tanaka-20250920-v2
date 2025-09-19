import argparse
import yaml
import os
import json
import torch
from .preprocess import get_datasets
from .train import train_model
from .evaluate import run_evaluation, analyze_and_plot_results

def run_experiment(config):
    """Runs the full experimental pipeline for a given configuration."""
    all_results = []
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    seeds = config.get('seeds', [42])
    noise_levels = config.get('noise_levels', [0])

    for seed in seeds:
        config['seed'] = seed
        torch.manual_seed(seed)
        
        for noise in noise_levels:
            print(f'\n--- Running Experiment: Seed={seed}, Noise Level={noise}% ---\n')
            current_config = config.copy()
            current_config['noise_level'] = noise

            # 1. Preprocessing
            train_ds, eval_ds = get_datasets(current_config, noise)

            # 2. Training
            model_path, model, tokenizer = train_model(current_config, train_ds, eval_ds)
            
            # 3. Evaluation
            metrics = run_evaluation(current_config, model, tokenizer)
            metrics['seed'] = seed
            all_results.append(metrics)
            
            # Clean up memory
            del model
            torch.cuda.empty_cache()

    # Aggregate, save, and plot results
    results_path = os.path.join(output_dir, 'aggregated_results.json')
    try:
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=4)
        print(f"\nAggregated results saved to {results_path}")
        print("--- AGGREGATED RESULTS ---")
        print(json.dumps(all_results, indent=2))
        print("------------------------")
    except IOError as e:
        print(f"Error saving results to {results_path}: {e}")

    analyze_and_plot_results(all_results, os.path.join(output_dir, 'images'))


def main():
    parser = argparse.ArgumentParser(description="Run R2-D2PO experiments.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke-test", action='store_true', help="Run a small-scale smoke test.")
    group.add_argument("--full-experiment", action='store_true', help="Run the full experiment.")
    args = parser.parse_args()

    if args.smoke_test:
        config_path = 'config/smoke_test.yaml'
    else:
        config_path = 'config/full_experiment.yaml'

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {config_path}: {e}")
        return

    run_experiment(config)

if __name__ == "__main__":
    main()
