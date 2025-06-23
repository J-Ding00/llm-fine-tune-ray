from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
import wandb
import yaml
import os
import torch
from dotenv import load_dotenv
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from scripts.llm_label_sample import label_batch_local


def tokenize_function(example, tokenizer, max_length):
    prompt_text = tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=True
    )

    # Create full training text
    full_text = prompt_text + example["output"]
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding='max_length',
    )
    prompt_tokenized = tokenizer(prompt_text, truncation=True, max_length=max_length, padding=False)
    prompt_len = len(prompt_tokenized["input_ids"])

    # Mask labels before the output
    labels = tokenized["input_ids"].copy()
    labels[:prompt_len] = [-100] * prompt_len
    tokenized["labels"] = labels

    return tokenized

def format_for_finetuning(example, criteria, criteria_format):
    # Create the structured format placeholder

    # Build the instruction
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

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    model_name = config['pretrain_model']['model']
    criteria = config['data_process']['criteria']
    criteria_format = {c: { "score": "<int>", "feedback": "<string>" } for c in criteria}
    out_dir = config['fine_tune']['output_dir']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    max_new_tokens = config['pretrain_model']['eval']["max_eval_tokens"]

    # lora_config = LoraConfig(
    #     r=8,
    #     lora_alpha=32,
    #     target_modules='all-linear',
    #     lora_dropout=0.05,
    #     bias="none",
    #     task_type=TaskType.CAUSAL_LM,
    # )

    lora_config = LoraConfig(
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # quantization_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.float16,
    # )

    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto",)
    model = get_peft_model(model, lora_config)

    # Load your dataset (e.g., JSONL -> Dataset)
    raw_dataset = load_dataset("json", data_files="data/train/clean_label_all_sample_train.jsonl")["train"]
    dataset = raw_dataset.map(lambda x: format_for_finetuning(x, criteria, criteria_format))

    # Tokenize
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_length=1024),
        remove_columns=dataset.column_names,
        batched=False,
    )

    raw_eval_dataset = load_dataset("json", data_files="data/val/clean_label_all_sample_val.jsonl")["train"]

    formatted_eval = raw_eval_dataset.map(lambda x: format_for_finetuning(x, criteria, criteria_format))

    tokenized_eval = formatted_eval.map(
        lambda x: tokenize_function(x, tokenizer, max_length=1024),
        remove_columns=formatted_eval.column_names,
        batched=False,
    )

    # Collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    load_dotenv()

    WANDB_API_KEY = os.getenv("WANDB_API_KEY")
    wandb.login(key=WANDB_API_KEY)
    run = wandb.init(
        project="LLM-fine-tune",
        name="qlora-qwen-finetune",
    )
    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        logging_steps=20,
        save_strategy='epoch',
        save_total_limit=3,
        warmup_ratio=0.05,
        fp16=False,
        bf16=True,
        dataloader_num_workers=4,
        prediction_loss_only=True,
        eval_strategy="steps",
        eval_steps=120,
        report_to="wandb",
        run_name="qlora-llama3-finetune",
        seed=42,
        gradient_checkpointing=True,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )


    import torch
    import gc
    torch.cuda.empty_cache()
    gc.collect()
    trainer.train()

    folders = [f for f in os.listdir(out_dir) if f.startswith("checkpoint") and os.path.isdir(os.path.join(out_dir, f))]
    os.makedirs('predictions', exist_ok=True)
    for i, f in enumerate(folders, start=1):
        base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", quantization_config=quantization_config)
        peft_model = PeftModel.from_pretrained(base_model, os.path.join(out_dir, f))
        torch.cuda.empty_cache()
        gc.collect()
        label_batch_local("data/test/clean_label_all_sample_test.jsonl", f'predictions/fine_tune_pred_epoch{i+1}.jsonl', criteria, peft_model, tokenizer, max_new_tokens)