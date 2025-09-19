
Input:
From the Hugging Face README provided in “# README,” extract and output only the Python code required for execution. Do not output any other information. In particular, if no implementation method is described, output an empty string.

# README
---
language:
- en
license: mit
size_categories:
- 10K<n<100K
task_categories:
- text-generation
pretty_name: UltraFeedback Binarized Preferences Cleaned
dataset_info:
  features:
  - name: source
    dtype: string
  - name: prompt
    dtype: string
  - name: chosen
    list:
    - name: content
      dtype: string
    - name: role
      dtype: string
  - name: chosen-rating
    dtype: float64
  - name: chosen-model
    dtype: string
  - name: rejected
    list:
    - name: content
      dtype: string
    - name: role
      dtype: string
  - name: rejected-rating
    dtype: float64
  - name: rejected-model
    dtype: string
  splits:
  - name: train
    num_bytes: 284937773
    num_examples: 60917
  download_size: 143257393
  dataset_size: 284937773
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
tags:
- dpo
- preference
- ultrafeedback
---

# UltraFeedback - Binarized using the Average of Preference Ratings (Cleaned)

This dataset represents a new iteration on top of [`argilla/ultrafeedback-binarized-preferences`](https://huggingface.co/argilla/ultrafeedback-binarized-preferences),
and is the **recommended and preferred dataset by Argilla to use from now on when fine-tuning on UltraFeedback**.

Read more about Argilla's approach towards UltraFeedback binarization at [`argilla/ultrafeedback-binarized-preferences/README.md`](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences/blob/main/README.md).

## Differences with `argilla/ultrafeedback-binarized-preferences`

Thanks to the recent issue identified by [AllenAI](https://huggingface.co/allenai) related to the TruthfulQA contamination within the
original UltraFeedback dataset due to some prompts being reused from the TruthfulQA dataset (used for benchmarking
in the [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) from HuggingFace H4), we also decided
to follow AllenAI's advice and remove those from the UltraFeedback dataset that we binarized using a completely different approach, which
implied using the average of the preference ratings rather than the critique overall score, as
[`HuggingFaceH4/ultrafeedback_binarized`](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) did.

Besides that, we also saw that not only the rows with the `source=truthful_qa` were contamined (for obvious reasons), but also some
coming from ShareGPT, so we also removed those doing a left join with both subsets from the [`truthful_qa`](https://huggingface.co/datasets/truthful_qa) dataset.

Additionally, we also modified the formatting to be aligned with both [`HuggingFaceH4/ultrafeedback_binarized`](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized),
and [`allenai/ultrafeedback_binarized_cleaned`](https://huggingface.co/datasets/allenai/ultrafeedback_binarized_cleaned) in order to ease
the integration within the [`huggingface/alignment-handbook`](https://github.com/huggingface/alignment-handbook) so that the formatting is standardized.

## Reproduce

<a target="_blank" href="https://colab.research.google.com/drive/1XR9P1St4yTNY0tjti_tIjm-yzP5Bfqc0?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

To reproduce the data processing combining both our approach and the suggestions from HuggingFace H4 w.r.t. the formatting and the ones from AllenAI to
remove the TruthfulQA contamination, feel free to run the attached Colab Notebook or just view it at [`notebook.ipynb`](./notebook.ipynb) within this repository.

From Argilla we encourage anyone out there to play around, investigate, and experiment with the data, and we firmly believe on open sourcing what we do, as
ourselves, as well as the whole community, benefit a lot from open source and we also want to give back.

## Citation

If you find this dataset is useful in your work, please cite the original UltraFeedback dataset: https://huggingface.co/datasets/openbmb/UltraFeedback

Additionally, you may also want to cite our work with Notus 7B, which lead the curation of the UltraFeedback dataset:

```bibtex
@misc{notus2023,
  author = {Alvaro Bartolome and Gabriel Martin and Daniel Vila},
  title = {Notus},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/argilla-io/notus}}
}
```

> Alphabetically ordered by last name due to equal contribution.
Output:
{
    "extracted_code": ""
}
