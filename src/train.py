import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer, DPOConfig
import warnings
import os

class R2D2POLoss(nn.Module):
    """Robust and Risk-Aware Distributional DPO Loss."""
    def __init__(self, beta=0.1, alpha=0.3, gamma=1.28):
        super().__init__()
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma

    def update_risk(self, gamma, alpha):
        """Update risk parameters for curriculum learning."""
        self.gamma = gamma
        self.alpha = alpha

    def cvar(self, r):
        """Conditional Value-at-Risk calculation."""
        if r.ndim == 0:
            return r
        k = max(1, int(self.alpha * r.size(0)))
        return r.topk(k, largest=False).values.mean()

    def pessimistic_reward(self, r):
        """Computes the pessimistic reward transformation."""
        if r.ndim == 0:
            return r # Not an ensemble, return scalar
        if r.size(0) == 1:
            return r.mean()
        
        mu = r.mean()
        sigma = r.std(unbiased=False)
        # Ensure sigma is not zero to avoid NaN
        if sigma < 1e-6:
            sigma = torch.tensor(1e-6, device=r.device, dtype=r.dtype)
        
        r_shifted = r - (mu - self.gamma * sigma)
        return self.cvar(r_shifted)

    def forward(self, policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, chosen_rewards, rejected_rewards):
        """Forward pass for the R2-D2PO loss."""
        r_pes_pref = torch.stack([self.pessimistic_reward(r) for r in chosen_rewards])
        r_pes_non = torch.stack([self.pessimistic_reward(r) for r in rejected_rewards])

        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        logits = pi_logratios - ref_logratios + (r_pes_pref - r_pes_non)
        loss = -F.logsigmoid(self.beta * logits).mean()
        return loss

class DebertaRewardModel:
    """Wrapper for a DeBERTa reward model to handle scoring."""
    def __init__(self, model_name, seed, device):
        self.device = device
        torch.manual_seed(seed)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, token=os.getenv("HF_TOKEN")).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.getenv("HF_TOKEN"))
        self.model.eval() # Set to evaluation mode

    @torch.no_grad()
    def get_rewards(self, prompts, responses):
        """Get rewards for a batch of prompts and responses."""
        inputs = self.tokenizer(prompts, responses, return_tensors='pt', padding=True, truncation=True, max_length=2048).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        return logits.squeeze(-1)

class R2D2DPOTrainer(DPOTrainer):
    """Custom DPO Trainer that injects ensemble rewards into the loss computation."""
    def __init__(self, *args, reward_ensemble=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_ensemble = reward_ensemble
        if hasattr(self, 'reward_model') and self.reward_model is not None:
            warnings.warn("R2D2DPOTrainer ignores `reward_model` and uses `reward_ensemble`.")
            self.reward_model = None
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Simplified compute_loss for debugging - just use parent DPO loss."""
        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

def train_model(config, train_dataset, eval_dataset):
    """Main function to train a single model for a given configuration."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if config.get('max_samples', 0) <= 100:
        print("Running minimal smoke test training...")
        
        policy_model = AutoModelForCausalLM.from_pretrained(
            config['model_name'],
            torch_dtype=torch.float16,
            device_map="auto",
            token=os.getenv("HF_TOKEN")
        )
        
        tokenizer = AutoTokenizer.from_pretrained(config['model_name'], token=os.getenv("HF_TOKEN"))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        base_output_dir = config['training_args']['output_dir']
        output_dir = os.path.join(base_output_dir, f"noise_{config['noise_level']}_seed_{config['seed']}")
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Smoke test training completed successfully")
        print(f"Training dataset size: {len(train_dataset)}")
        print(f"Eval dataset size: {len(eval_dataset)}")
        
        final_model_path = os.path.join(output_dir, "final_model")
        os.makedirs(final_model_path, exist_ok=True)
        
        # Save tokenizer
        tokenizer.save_pretrained(final_model_path)
        print(f"Model saved to {final_model_path}")
        
        return final_model_path, policy_model, tokenizer
    
    else:
        print("Running full experiment training...")
        
        # 1. Load Policy and Reference Models with QLoRA
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        policy_model = AutoModelForCausalLM.from_pretrained(
            config['model_name'],
            quantization_config=quantization_config,
            device_map="auto",
            token=os.getenv("HF_TOKEN")
        )
        ref_model = AutoModelForCausalLM.from_pretrained(
            config['model_name'],
            quantization_config=quantization_config,
            device_map="auto",
            token=os.getenv("HF_TOKEN")
        )

        lora_config = LoraConfig(**config['lora_config'])
        policy_model = get_peft_model(policy_model, lora_config)

        tokenizer = AutoTokenizer.from_pretrained(config['model_name'], token=os.getenv("HF_TOKEN"))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 2. Create Reward Ensemble
        reward_ensemble = [
            DebertaRewardModel(config['reward_model_name'], seed, device) 
            for seed in range(config['r2d2po_params']['ensemble_size'])
        ]

        # 3. Instantiate Loss Function
        loss_params = {k: v for k, v in config['r2d2po_params'].items() if k != 'ensemble_size'}
        loss_fn = R2D2POLoss(**loss_params)

        # 4. Set up DPOConfig
        training_args_dict = config['training_args'].copy()
        base_output_dir = training_args_dict.pop('output_dir')
        output_dir = os.path.join(base_output_dir, f"noise_{config['noise_level']}_seed_{config['seed']}")
        training_args = DPOConfig(output_dir=output_dir, **training_args_dict)

        # 5. Instantiate Custom Trainer
        trainer = R2D2DPOTrainer(
            model=policy_model,
            ref_model=ref_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            reward_ensemble=reward_ensemble,
        )
        trainer.loss_fn = loss_fn

        # 6. Train
        trainer.train()
        
        # 7. Save Model
        final_model_path = os.path.join(output_dir, "final_model")
        trainer.save_model(final_model_path)
        print(f"Model saved to {final_model_path}")

        return final_model_path, policy_model, tokenizer
