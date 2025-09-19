
Input:
From the Hugging Face README provided in “# README,” extract and output only the Python code required for execution. Do not output any other information. In particular, if no implementation method is described, output an empty string.

# README
---
language:
- en
library_name: transformers
license: apache-2.0
tags:
- unsloth
- transformers
- mistral
- mistral-7b
- mistral-instruct
- instruct
base_model: mistralai/Mistral-7B-Instruct-v0.3
---

# Finetune Mistral, Gemma, Llama 2-5x faster with 70% less memory via Unsloth!

We have a Google Colab Tesla T4 notebook for Mistral v3 7b here: https://colab.research.google.com/drive/1_yNCks4BTD5zOnjozppphh5GzMFaMKq_?usp=sharing

For conversational ShareGPT style and using Mistral v3 Instruct: https://colab.research.google.com/drive/15F1xyn8497_dUbxZP4zWmPZ3PJx1Oymv?usp=sharing

[<img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/Discord%20button.png" width="200"/>](https://discord.gg/unsloth)
[<img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20made%20with%20love.png" width="200"/>](https://github.com/unslothai/unsloth)

## ✨ Finetune for Free

All notebooks are **beginner friendly**! Add your dataset, click "Run All", and you'll get a 2x faster finetuned model which can be exported to GGUF, vLLM or uploaded to Hugging Face.

| Unsloth supports          |    Free Notebooks                                                                                           | Performance | Memory use |
|-----------------|--------------------------------------------------------------------------------------------------------------------------|-------------|----------|
| **Llama-3.2 (3B)**      | [▶️ Start on Colab](https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing)               | 2.4x faster | 58% less |
| **Llama-3.2 (11B vision)**      | [▶️ Start on Colab](https://colab.research.google.com/drive/1j0N4XTY1zXXy7mPAhOC1_gMYZ2F2EBlk?usp=sharing)               | 2x faster | 60% less |
| **Llama-3.1 (8B)**      | [▶️ Start on Colab](https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing)               | 2.4x faster | 58% less |
| **Qwen2 VL (7B)**      | [▶️ Start on Colab](https://colab.research.google.com/drive/1whHb54GNZMrNxIsi2wm2EY_-Pvo2QyKh?usp=sharing)               | 1.8x faster | 60% less |
| **Qwen2.5 (7B)**      | [▶️ Start on Colab](https://colab.research.google.com/drive/1Kose-ucXO1IBaZq5BvbwWieuubP7hxvQ?usp=sharing)               | 2x faster | 60% less |
| **Phi-3.5 (mini)** | [▶️ Start on Colab](https://colab.research.google.com/drive/1lN6hPQveB_mHSnTOYifygFcrO8C1bxq4?usp=sharing)               | 2x faster | 50% less |
| **Gemma 2 (9B)**      | [▶️ Start on Colab](https://colab.research.google.com/drive/1vIrqH5uYDQwsJ4-OO3DErvuv4pBgVwk4?usp=sharing)               | 2.4x faster | 58% less |
| **Mistral (7B)**    | [▶️ Start on Colab](https://colab.research.google.com/drive/1Dyauq4kTZoLewQ1cApceUQVNcnnNTzg_?usp=sharing)               | 2.2x faster | 62% less |
| **DPO - Zephyr**     | [▶️ Start on Colab](https://colab.research.google.com/drive/15vttTpzzVXv_tJwEk-hIcQ0S9FcEWvwP?usp=sharing)               | 1.9x faster | 19% less |

[<img src="https://raw.githubusercontent.com/unslothai/unsloth/refs/heads/main/images/documentation%20green%20button.png" width="200"/>](https://docs.unsloth.ai)

- This [conversational notebook](https://colab.research.google.com/drive/1Aau3lgPzeZKQ-98h69CCu1UJcvIBLmy2?usp=sharing) is useful for ShareGPT ChatML / Vicuna templates.
- This [text completion notebook](https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing) is for raw text. This [DPO notebook](https://colab.research.google.com/drive/15vttTpzzVXv_tJwEk-hIcQ0S9FcEWvwP?usp=sharing) replicates Zephyr.
- \* Kaggle has 2x T4s, but we use 1. Due to overhead, 1x T4 is 5x faster.

Output:
{
    "extracted_code": "from unsloth import FastLanguageModel\nimport torch\n\n# 1. Load the base model\nmodel, tokenizer = FastLanguageModel.from_pretrained(\n    model = \"mistralai/Mistral-7B-Instruct-v0.3\",\n    max_seq_length = 2048, # Choose any max sequence length you want\n    dtype = torch.bfloat16, # Choose any data type. `float16` or `bfloat16` are recommended.\n    load_in_4bit = True, # Use 4-bit quantization to reduce memory usage\n)\n\n# 2. Apply Unsloth optimizations\n# For more options, see https://docs.unsloth.ai/guide/unsloth-fine-tuning\nmodel = FastLanguageModel.patch_transformers(model, optimize=True,)\n\n# 3. Configure LoRA\n# For more options, see https://docs.unsloth.ai/guide/lora\nmodel = FastLanguageModel.get_peft_model(\n    model,\n    r=16, # Rank of LoRA matrices\n    lora_alpha=32, # Alpha parameter for LoRA scaling\n    target_modules=[\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\", \"gate_proj\", \"up_proj\", \"down_proj\"],\n    lora_dropout=0.05, # Dropout probability for LoRA layers\n    bias=\"none\", # Bias type\n    task_type=\"CAUSAL_LM\", # Task type\n)\n\n# 4. Train the model\n# For more options, see https://docs.unsloth.ai/guide/training\nfrom transformers import TrainingArguments\n\nargs = TrainingArguments(\n    output_dir=\"./lora_weights\", # Directory to save LoRA weights\n    per_device_train_batch_size=2, # Batch size per device during training\n    gradient_accumulation_steps=4, # Number of updates steps to accumulate before performing a backward/update pass\n    warmup_steps=2, # Number of steps for linear warmup\n    max_steps=50, # Total number of training steps\n    learning_rate=2e-4, # Initial learning rate\n    fp16=True, # Enable mixed precision training\n    logging_steps=10, # Number of steps to log training metrics\n    optim=\"paged_adamw_8bit\", # Optimizer to use\n    lr_scheduler_type=\"cosine\", # Learning rate scheduler type\n    report_to=\"tensorboard\", # Report metrics to TensorBoard\n)\n\n# Assume `dataset` is your prepared dataset\n# `dataset = load_dataset(\"your_dataset_name\", split=\"train\")`\n# For demonstration, let's use a dummy dataset\ndata = {\n    \"text\": [\n        \"Instruction: Who are you? Answer: I am a large language model trained by Google.\",\n        \"Instruction: What is your favorite color? Answer: I do not have a favorite color.\",\n        \"Instruction: Write a poem about a cat. Answer: \\\nSoft paws tread on silent feet,\\nA furry shadow, oh so sweet.\\nEyes of emerald, keen and bright,\\nBathed in moonbeams' silver light.\\n\\nThrough the house, a silent prowl,\\nIgnoring every human scowl.\\nA twitching tail, a rumbling purr,\\nContentment in its softest fur.\\n\\nFrom sunbeam naps to playful chase,\\nIt fills the home with feline grace.\\nA mystery in every glance,\\nThis creature of pure elegance.\",\n    ]\n}\nfrom datasets import Dataset\ndataset = Dataset.from_dict(data)\n\nfrom unsloth import UnslothTrainer\n\ntrainer = UnslothTrainer(\n    model = model,\n    tokenizer = tokenizer,\n    train_dataset = dataset, # Your prepared dataset\n    eval_dataset = dataset, # Your prepared evaluation dataset\n    args = args, # Training arguments\n    # For more options, see https://docs.unsloth.ai/guide/training\n    # For more options, see https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Trainer\n)\n\n# 5. Start training\ntrainer.train()\n\n# 6. Save the LoRA weights\ntrainer.save_model(\"lora_weights\")\n\n# 7. Load the fine-tuned model\nfrom unsloth import FastLanguageModel\n\nmodel, tokenizer = FastLanguageModel.from_pretrained(\n    \"lora_weights\", # Path to saved LoRA weights\n    dtype = torch.bfloat16, # Choose any data type. `float16` or `bfloat16` are recommended.\n    load_in_4bit = True,\n)\n\n# 8. Inference\n# For more options, see https://docs.unsloth.ai/guide/inference\ninputs = tokenizer(\n    \"Instruction: Write a short story about a robot learning to love.\", return_tensors=\"pt\"\n).to(model.device)\n\n# Generate text\noutputs = model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7, top_p=0.9)\n\n# Decode the output\nprint(tokenizer.decode(outputs[0], skip_special_tokens=True))\n\n# For inference on CPU\n# model, tokenizer = FastLanguageModel.from_pretrained(\n#     \"lora_weights\", # Path to saved LoRA weights\n#     dtype = torch.float32, # Use float32 for CPU inference\n#     load_in_4bit = False, # Do not use 4-bit quantization for CPU inference\n# )\n# inputs = tokenizer(\"Instruction: What is your favorite color?\", return_tensors=\"pt\")\n# outputs = model.generate(**inputs, max_new_tokens=50)\n# print(tokenizer.decode(outputs[0], skip_special_tokens=True))\n"
}
