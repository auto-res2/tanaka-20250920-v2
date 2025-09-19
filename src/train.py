import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer
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
        if self.reward_model is not None:
            warnings.warn("R2D2DPOTrainer ignores `reward_model` and uses `reward_ensemble`.")
            self.reward_model = None
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Override compute_loss to use the reward ensemble."""
        # Get log probabilities from policy and reference models
        policy_chosen_logps, policy_rejected_logps, policy_chosen_logits, policy_rejected_logits = self.get_logps(model, inputs)

        # Use pre-computed reference logps
        reference_chosen_logps = inputs["reference_chosen_logps"]
        reference_rejected_logps = inputs["reference_rejected_logps"]
        
        # Get rewards from the ensemble
        prompts = [self.tokenizer.decode(p, skip_special_tokens=True) for p in inputs["prompt_input_ids"]]
        chosens = [self.tokenizer.decode(c, skip_special_tokens=True) for c in inputs["chosen_input_ids"]]
        rejecteds = [self.tokenizer.decode(r, skip_special_tokens=True) for r in inputs["rejected_input_ids"]]
        
        chosen_rewards_list = []
        rejected_rewards_list = []
        for rm in self.reward_ensemble:
            chosen_scores = rm.get_rewards(prompts, chosens)
            rejected_scores = rm.get_rewards(prompts, rejecteds)
            chosen_rewards_list.append(chosen_scores)
            rejected_rewards_list.append(rejected_scores)

        # Stack rewards along a new dimension for the ensemble
        chosen_rewards = torch.stack(chosen_rewards_list, dim=1)
        rejected_rewards = torch.stack(rejected_rewards_list, dim=1)

        # Compute loss using the custom loss function
        loss = self.loss_fn(
            policy_chosen_logps, policy_rejected_logps,
            reference_chosen_logps, reference_rejected_logps,
            chosen_rewards, rejected_rewards
        )
        
        # Keep track of metrics
        chosen_rewards_mean = chosen_rewards.mean()
        rejected_rewards_mean = rejected_rewards.mean()
        self.store_metrics({'rewards/chosen': chosen_rewards_mean, 'rewards/rejected': rejected_rewards_mean})
        
        if return_outputs:
            return loss, {'chosen_rewards': chosen_rewards_mean, 'rejected_rewards': rejected_rewards_mean}
        return loss

def train_model(config, train_dataset, eval_dataset):
    """Main function to train a single model for a given configuration."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Policy and Reference Models with QLoRA
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    
    policy_model = AutoModelForCausalLM.from_pretrained(
        config['model_name'],
        quantization_config=quantization_config,
        device_map="auto",
        token=os.getenv("HF_TOKEN")
    )

    # It's common to use the same model as ref, without PEFT adapters
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
    loss_fn = R2D2POLoss(**config['r2d2po_params'])

    # 4. Set up TrainingArguments
    training_args_dict = config['training_args']
    output_dir = os.path.join(training_args_dict['output_dir'], f"noise_{config['noise_level']}_seed_{config['seed']}")
    training_args = TrainingArguments(output_dir=output_dir, **training_args_dict)

    # 5. Instantiate Custom Trainer
    trainer = R2D2DPOTrainer(
        model=policy_model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        loss_fn=loss_fn,
        reward_ensemble=reward_ensemble,
        max_prompt_length=2048,
        max_length=4096,
    )

    # 6. Train
    trainer.train()
    
    # 7. Save Model
    final_model_path = os.path.join(output_dir, "final_model")
    trainer.save_model(final_model_path)
    print(f"Model saved to {final_model_path}")

    return final_model_path, policy_model, tokenizer
