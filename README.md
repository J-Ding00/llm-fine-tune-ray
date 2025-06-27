# Fine-Tuning LLM for Structured Speech Feedback

This project fine-tunes a 3B parameter language model [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) to generate **structured JSON feedback** (trait-based scores and explanations) for speech transcripts. It combines hard-label distillation from GPT-4o with scalable, distributed training and evaluation powered by Ray.

- **Data:** ~1,200 OpenWebText samples, labeled and augmented by GPT-4o/GPT-4o-mini
- **Model:** Qwen2.5-3B-Instruct
- **Labeling:** Hard-label distillation from GPT-4o ("LLM as a judge")
- **Compute:** NVIDIA L4 GPUs, Ray for distributed training & evaluation

## Key Results
- **MSE:** 8.8 → 1.2
- **JSON validity:** 98% → 100%
- **BERTScore:** 0.90 → 0.93
- **Consistent improvement** across traits (formality, empathy, etc.)

**Distributed with Ray:**  
- 2x faster training with 2 GPUs (Ray Train)
- 3x faster batch evaluation (Ray Data)

[See detailed results and scripts → `docs/DETAILS.md`](docs/DETAILS.md)