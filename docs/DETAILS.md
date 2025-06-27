# Fine-Tuning Details and Ray-Distributed Workflow

## Overview

This project fine-tunes a 3B parameter language model [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) to generate structured, per-trait feedback on speech transcripts.  
**Labels are hard-distilled from GPT-4o outputs.** Data includes synthetic augmentation for low-score minority classes.

- **Data:** OpenWebText (1,000 labeled + 200 GPT-4o-mini augmented)
- **Labeling:** GPT-4o as a judge (hard labels)
- **Fine-tune:** QLoRA, chat-format, single/multi-GPU (Ray)
- **Training Instance:** NVIDIA L4

---

## Evaluation

Each model (pretrained & fine-tuned) is evaluated by:
- Generating JSON feedback for test set
- Comparing outputs to GPT-4o ground truth

**Metrics:**
- JSON validity rate (parseable outputs; production-critical)
- MSE (mean squared error on trait scores)
- BERTScore (semantic similarity on text feedback)

---

## Results

| Metric         | Pretrained | Fine-tuned (Epoch 3)|
|:---------------|:----------:|:-----------------:|
| Avg MSE        |   8.8      |     1.2           |
| JSON Validity  |  98%      |   100%            |
| BERTScore (F1) |   0.90     |    0.93           |

---

## Ray-Distributed Experiments

- **Training:**  
  - 1× L4 GPU: 36m  
  - 2× L4 GPUs (Ray Train): 18m  
  - *Near-linear scaling for LoRA fine-tune workflow*

- **Batch Evaluation:**  
  - Classic Transformers (batch=20): ~7m  
  - Ray Data LLM: ~2.5m (up to 3× speedup)

---

## Notes

- Only JSON-valid outputs counted in metrics.
- All data/workflows/scripts provided in [`scripts/`](../scripts/) and [`eval/`](../eval/).

> More detailed training metrics and graphs available at [Weights & Biases](https://wandb.ai/jcdingjobs-independent/LLM-fine-tune/runs/11k3ev24?nw=nwuserjcdingjobs).