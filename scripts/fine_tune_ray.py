import os
import gc
import yaml
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import get_peft_model, LoraConfig, TaskType

import ray
from ray.train import ScalingConfig, RunConfig
from ray.train.torch import TorchTrainer
import ray.train.huggingface.transformers

def tokenize_function(example, tokenizer, max_length):
    prompt_text = tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=True
    )
    full_text = prompt_text + example["output"]
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding='max_length',
    )
    prompt_tokenized = tokenizer(prompt_text, truncation=True, max_length=max_length, padding=False)
    prompt_len = len(prompt_tokenized["input_ids"])
    labels = tokenized["input_ids"].copy()
    labels[:prompt_len] = [-100] * prompt_len
    tokenized["labels"] = labels
    return tokenized

def format_for_finetuning(example, criteria, criteria_format):
    instruction = f"""
    You are an expert speech evaluator. Given a transcript of a speaker's message, assess the communication quality in terms of the following traits:
    {criteria}

    For each trait, assign a score from 1 (poor) to 10 (excellent). Be honest and critical in your assessment without bias toward high scores. 
    Then provide a short explanation (1–2 sentences) justifying your score for that trait.

    The final output must be a JSON string with the exact format below:
    {criteria_format}
    """.strip()
    transcript = f"Transcript:\n{example['text']}"
    messages = [
        {"role": "system", "content": f"{instruction}"},
        {"role": "user", "content": f"{transcript}"}
    ]
    return {
        "messages": messages,
        "output": example["label"]
    }

def train_func():
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    model_name = config['llm_label']['model']
    criteria = config['criteria']
    criteria_format = {c: {"score": "<int>", "feedback": "<string>"} for c in criteria}
    ft_config = config['fine_tune']
    data_path = ft_config['data_path']

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    lora_config = LoraConfig(
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
    model = get_peft_model(model, lora_config)

    # Load and preprocess datasets
    raw_dataset = load_dataset("json", data_files=data_path['train'])["train"]
    dataset = raw_dataset.map(lambda x: format_for_finetuning(x, criteria, criteria_format))
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_length=1024),
        remove_columns=dataset.column_names,
        batched=False,
    )

    raw_eval_dataset = load_dataset("json", data_files=data_path['val'])["train"]
    formatted_eval = raw_eval_dataset.map(lambda x: format_for_finetuning(x, criteria, criteria_format))
    tokenized_eval = formatted_eval.map(
        lambda x: tokenize_function(x, tokenizer, max_length=1024),
        remove_columns=formatted_eval.column_names,
        batched=False,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # run = wandb.init(
    #     project="LLM-fine-tune",
    #     name="lora-qwen-ray",
    # )
    training_args = TrainingArguments(
        output_dir="test_trainer",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        logging_steps=20,
        save_strategy='epoch',
        save_total_limit=3,
        warmup_ratio=0.05,
        fp16=False,
        bf16=True,
        dataloader_num_workers=2,
        prediction_loss_only=True,
        eval_strategy="steps",
        eval_steps=120,
        weight_decay=0.01,
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Add Ray callback and prepare trainer for distributed training
    callback = ray.train.huggingface.transformers.RayTrainReportCallback()
    trainer.add_callback(callback)
    trainer = ray.train.huggingface.transformers.prepare_trainer(trainer)

    # Optional: clear CUDA cache before training
    torch.cuda.empty_cache()
    gc.collect()

    trainer.train()

if __name__ == "__main__":
    import ray
    from ray.train import ScalingConfig
    from ray.train.torch import TorchTrainer

    os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"
    ray.init(runtime_env={"working_dir": "."})

    # Launch training on 2 workers, each with 1 GPU (adjust as needed)
    ray_trainer = TorchTrainer(
        train_func,
        scaling_config=ScalingConfig(num_workers=2, use_gpu=True),
        run_config=RunConfig(storage_path='/mnt/cluster_storage/llm-fine-tune-ray/checkpoints_ray1')
    )
    result = ray_trainer.fit()