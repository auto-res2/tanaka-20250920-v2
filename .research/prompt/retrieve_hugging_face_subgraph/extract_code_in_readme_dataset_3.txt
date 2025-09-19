
Input:
From the Hugging Face README provided in “# README,” extract and output only the Python code required for execution. Do not output any other information. In particular, if no implementation method is described, output an empty string.

# README
---
license: mit
configs:
- config_name: default
  data_files:
  - split: train_sft
    path: data/train_sft-*
  - split: test_sft
    path: data/test_sft-*
  - split: train_gen
    path: data/train_gen-*
  - split: test_gen
    path: data/test_gen-*
  - split: train_prefs
    path: data/train_prefs-*
  - split: test_prefs
    path: data/test_prefs-*
dataset_info:
  features:
  - name: prompt
    dtype: string
  - name: prompt_id
    dtype: string
  - name: chosen
    list:
    - name: content
      dtype: string
    - name: role
      dtype: string
  - name: rejected
    list:
    - name: content
      dtype: string
    - name: role
      dtype: string
  - name: messages
    list:
    - name: content
      dtype: string
    - name: role
      dtype: string
  - name: score_chosen
    dtype: float64
  - name: score_rejected
    dtype: float64
  - name: source
    dtype: string
  splits:
  - name: train_sft
    num_bytes: 393926052.7984401
    num_examples: 60829
  - name: test_sft
    num_bytes: 6230841.363636363
    num_examples: 985
  - name: train_gen
    num_bytes: 314344767.49216783
    num_examples: 60829
  - name: test_gen
    num_bytes: 4982506.090909091
    num_examples: 985
  - name: train_prefs
    num_bytes: 393926052.7984401
    num_examples: 60829
  - name: test_prefs
    num_bytes: 12672623.615773508
    num_examples: 1964
  download_size: 629736515
  dataset_size: 1126082844.1593668
---
# Dataset Card for "ultrafeedback_binarized_cleaned"

**Update 1/12/2023**: I've removed examples identified as faulty by Argilla - see [their awesome work](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences) for more details.

This is a version of the [UltraFeedback binarized dataset](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) but with TruthfulQA prompts removed and source annotations added (so you can filter out samples from different sources yourself if you want!).

Please see the [binarized dataset card for more information](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized), or the [original UltraFeedback dataset card](https://huggingface.co/datasets/openbmb/UltraFeedback).
Output:
{
    "extracted_code": ""
}
