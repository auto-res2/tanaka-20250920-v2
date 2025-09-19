
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
- gemma
- gemma-7b
- bnb
---

# Finetune Mistral, Gemma, Llama 2-5x faster with 70% less memory via Unsloth!

We have a Google Colab Tesla T4 notebook for Gemma 7b here: https://colab.research.google.com/drive/10NbwlsRChbma1v55m8LAPYG15uQv6HLo?usp=sharing

[<img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/Discord%20button.png" width="200"/>](https://discord.gg/u54VK8m8tk)
[<img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/buy%20me%20a%20coffee%20button.png" width="200"/>](https://ko-fi.com/unsloth)
[<img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20made%20with%20love.png" width="200"/>](https://github.com/unslothai/unsloth)

## ✨ Finetune for Free

All notebooks are **beginner friendly**! Add your dataset, click "Run All", and you'll get a 2x faster finetuned model which can be exported to GGUF, vLLM or uploaded to Hugging Face.

| Unsloth supports          |    Free Notebooks                                                                                           | Performance | Memory use |
|-----------------|--------------------------------------------------------------------------------------------------------------------------|-------------|----------|
| **Gemma 7b**      | [▶️ Start on Colab](https://colab.research.google.com/drive/10NbwlsRChbma1v55m8LAPYG15uQv6HLo?usp=sharing)               | 2.4x faster | 58% less |
| **Mistral 7b**    | [▶️ Start on Colab](https://colab.research.google.com/drive/1Dyauq4kTZoLewQ1cApceUQVNcnnNTzg_?usp=sharing)               | 2.2x faster | 62% less |
| **Llama-2 7b**      | [▶️ Start on Colab](https://colab.research.google.com/drive/1lBzz5KeZJKXjvivbYvmGarix9Ao6Wxe5?usp=sharing)               | 2.2x faster | 43% less |
| **TinyLlama**  | [▶️ Start on Colab](https://colab.research.google.com/drive/1AZghoNBQaMDgWJpi4RbffGM1h6raLUj9?usp=sharing)              | 3.9x faster | 74% less |
| **CodeLlama 34b** A100   | [▶️ Start on Colab](https://colab.research.google.com/drive/1y7A0AxE3y8gdj4AVkl2aZX47Xu3P1wJT?usp=sharing)              | 1.9x faster | 27% less |
| **Mistral 7b** 1xT4  | [▶️ Start on Kaggle](https://www.kaggle.com/code/danielhanchen/kaggle-mistral-7b-unsloth-notebook) | 5x faster\* | 62% less |
| **DPO - Zephyr**     | [▶️ Start on Colab](https://colab.research.google.com/drive/15vttTpzzVXv_tJwEk-hIcQ0S9FcEWvwP?usp=sharing)               | 1.9x faster | 19% less |

- This [conversational notebook](https://colab.research.google.com/drive/1Aau3lgPzeZKQ-98h69CCu1UJcvIBLmy2?usp=sharing) is useful for ShareGPT ChatML / Vicuna templates.
- This [text completion notebook](https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing) is for raw text. This [DPO notebook](https://colab.research.google.com/drive/15vttTpzzVXv_tJwEk-hIcQ0S9FcEWvwP?usp=sharing) replicates Zephyr.
- \* Kaggle has 2x T4s, but we use 1. Due to overhead, 1x T4 is 5x faster.

Output:
{
    "extracted_code": "from unsloth import FastLanguageModel\nimport torch\n\nmodel, tokenizer = FastLanguageModel.from_pretrained(\n    model = \"unsloth/gemma-7b-bnb-4bit\",\n    max_seq_length = 2048,\n    torch_dtype = torch.float16,\n    load_in_4bit = True,\n)\n\nfrom transformers import TrainingArguments\n\nargs = TrainingArguments(\n    output_dir=\"./gemma_7b_results\",\n    per_device_train_batch_size = 2,\n    gradient_accumulation_steps = 4,\n    warmup_steps = 5,\n    max_steps = 60,\n    fp16 = True,\n    logging_steps = 1,\n    report_to = \"tensorboard\",\n)\n\nfrom datasets import load_dataset\n\ndata = load_dataset(\"abiyas/wiki_bio\", split=\"train\")\ndata = data.map(lambda x: tokenizer(x[\"text\"], max_length=512, truncation=True, padding=\"max_length\"), batched=True, remove_columns=[\"text\"])\n\nfrom unsloth import UnslothTrainer\n\ntrainer = UnslothTrainer(\n    model = model,\n    tokenizer = tokenizer,\n    train_dataset = data,\n    eval_dataset = data,\n    args = args,\n    data_collator = None # For simplicity, we'll use the default data collator.\n)\n\ntrainer.train()\n\n# Save to Hugging Face\ntrainer.push_to_hub(\"your_username/gemma_7b_finetuned\")\n\n# Save locally\ntrainer.save_model(\"gemma_7b_finetuned\")"
}
