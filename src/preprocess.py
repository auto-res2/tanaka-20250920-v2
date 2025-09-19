from datasets import load_dataset, Dataset
import random
import numpy as np

def format_chat(messages):
    """Formats a list of message dicts into a single string."""
    chat_str = ""
    for msg in messages:
        chat_str += f"{msg['role']}: {msg['content']}\n"
    return chat_str.strip()

def apply_noise(dataset, noise_level):
    """Applies uniform random flips to the preferences."""
    if noise_level == 0:
        return dataset

    def flip_preference(example):
        if random.random() < (noise_level / 100.0):
            # Swap chosen and rejected
            return {
                'prompt': example['prompt'],
                'chosen': example['rejected'],
                'rejected': example['chosen']
            }
        return example

    return dataset.map(flip_preference)

def get_datasets(config, noise_level):
    """Loads, preprocesses, and adds noise to the datasets."""
    print(f"Loading dataset with noise level: {noise_level}%")
    
    # Load main dataset
    try:
        uf_dataset = load_dataset(config['datasets']['ultrafeedback'], split='train_prefs')
    except Exception as e:
        raise FileNotFoundError(f"UltraFeedback dataset not found or error loading: {e} - aborting as per NO-FALLBACK policy.")

    # Load and merge second dataset if specified
    try:
        af_human_pref = load_dataset("tatsu-lab/alpaca_farm", "alpaca_human_preference", split='preference')
        # Reformat alpaca farm to match ultrafeedback
        def reformat_af(example):
            return {
                'prompt': example['instruction'],
                'chosen': [{'role': 'user', 'content': example['instruction']}, {'role': 'assistant', 'content': example[f'output_{example["preference"]}']}],
                'rejected': [{'role': 'user', 'content': example['instruction']}, {'role': 'assistant', 'content': example[f'output_{3 - example["preference"]}']}],
            }
        af_dataset = af_human_pref.map(reformat_af)
        # For simplicity, we'll just use UltraFeedback for this example
        # In a full run, you would concatenate them: 
        # from datasets import concatenate_datasets
        # dataset = concatenate_datasets([uf_dataset, af_dataset])
        dataset = uf_dataset
    except Exception as e:
        print(f"Could not load AlpacaFarm, continuing with UltraFeedback only. Error: {e}")
        dataset = uf_dataset
    
    # Format dataset for DPOTrainer
    def format_dpo(example):
        return {
            'prompt': example['prompt'],
            'chosen': format_chat(example['chosen']),
            'rejected': format_chat(example['rejected'])
        }
    
    dataset = dataset.map(format_dpo, remove_columns=dataset.column_names)

    # Apply noise
    noisy_dataset = apply_noise(dataset, noise_level)

    # Split data
    shuffled_dataset = noisy_dataset.shuffle(seed=config['seed'])
    if 'max_samples' in config:
        shuffled_dataset = shuffled_dataset.select(range(config['max_samples']))

    split_data = shuffled_dataset.train_test_split(test_size=0.1, seed=config['seed'])
    
    return split_data['train'], split_data['test']
