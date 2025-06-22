import json
import yaml
import os
from tqdm import tqdm
# from utils.utils import ray_data_step_logger
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def build_prompt(text, criteria, criteria_format):
    instruction = f"""
    You are an expert speech evaluator. Given a transcript of a speaker's message, assess the communication quality in terms of the following traits:
    {criteria}

    For each trait, assign a score from 1 (poor) to 10 (excellent). Be honest and critical in your assessment without bias toward high scores. 
    Then provide a short explanation (1–2 sentences) justifying your score for that trait.

    The final output must be a JSON string with the exact format below:
    {criteria_format}
    """.strip()

    transcript = f"Transcript:\n{text}"
    messages = [
        {"role": "system", "content": f"{instruction}"},
        {"role": "user", "content": f"{transcript}"}
    ]
    return messages

def generate_local_feedback(model, tokenizer, text, criteria, criteria_format):
    messages = build_prompt(text, criteria, criteria_format)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.2
    )
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return decoded

def label_batch_local(input_path, output_path, criteria, model_name, tokenizer):
    # Local batch eval
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    with open(input_path, "r") as f:
        lines = [json.loads(l) for l in f if l.strip()]
    criteria_format = {c: {"score": "<int>", "feedback": "<string>"} for c in criteria}
    with open(output_path, "w") as fout:
        for example in tqdm(lines, desc="Labeling samples (local)"):
            transcript = example["text"]
            try:
                feedback = generate_local_feedback(model, tokenizer, transcript, criteria, criteria_format)
                labeled = {"text": transcript, "label": feedback}
                fout.write(json.dumps(labeled, ensure_ascii=False) + "\n")
            except Exception:
                print("[Error] Skipping sample")

def label_batch_local_lora(input_path, output_path, criteria, model_name, adapter_path, tokenizer):
    from peft import PeftModel
    import torch
    base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    with open(input_path, "r") as f:
        lines = [json.loads(l) for l in f if l.strip()]
    criteria_format = {c: {"score": "<int>", "feedback": "<string>"} for c in criteria}
    with open(output_path, "w") as fout:
        for example in tqdm(lines, desc="Labeling samples (local + LoRA)"):
            transcript = example["text"]
            try:
                feedback = generate_local_feedback(model, tokenizer, transcript, criteria, criteria_format)
                labeled = {"text": transcript, "label": feedback}
                fout.write(json.dumps(labeled, ensure_ascii=False) + "\n")
            except Exception:
                print("[Error] Skipping sample")

def label_batch_ray(input_path, output_path, criteria, model_name, tokenizer):
    import ray
    from ray.data.llm import build_llm_processor, vLLMEngineProcessorConfig
    ray.init()
    criteria_format = {c: {"score": "<int>", "feedback": "<string>"} for c in criteria}
    with open(input_path, "r") as f:
        lines = [json.loads(l) for l in f if l.strip()]
    prompts = [
        {"prompt": tokenizer.apply_chat_template(
            build_prompt(example["text"], criteria, criteria_format),
            tokenize=False,
            add_generation_prompt=True
        ), "text": example["text"]} for example in lines
    ]
    processor_config = vLLMEngineProcessorConfig(
        model_source=model_name,
        engine_kwargs={
            "enable_chunked_prefill": True,
            "max_num_batched_tokens": 4096,
            "max_model_len": 4096,
        },
        concurrency=1,
        batch_size=8,
    )
    processor = build_llm_processor(
        processor_config,
        preprocess=lambda row: dict(
            messages=[
                {"role": "system", "content": "You are an expert speech evaluator."},
                {"role": "user", "content": row["prompt"]},
            ],
            sampling_params=dict(
                temperature=0.2,
                max_tokens=512,
            )
        ),
        postprocess=lambda row: dict(
            answer=row["generated_text"],
            text=row["text"],
        ),
    )
    ds = ray.data.from_items(prompts)
    ds = processor(ds)
    ds = ds.materialize()
    with open(output_path, "w") as fout:
        for out in ds.take_all():
            fout.write(json.dumps({"text": out["text"], "label": out["answer"]}, ensure_ascii=False) + "\n")
    ray.shutdown()

def label_batch_ray_lora(input_path, output_path, criteria, model_name, tokenizer, adapter_path, lora_rank=8):
    import ray
    from ray.data.llm import build_llm_processor, vLLMEngineProcessorConfig
    ray.init()
    criteria_format = {c: {"score": "<int>", "feedback": "<string>"} for c in criteria}
    with open(input_path, "r") as f:
        lines = [json.loads(l) for l in f if l.strip()]
    prompts = [
        {"prompt": tokenizer.apply_chat_template(
            build_prompt(example["text"], criteria, criteria_format),
            tokenize=False,
            add_generation_prompt=True
        ), "text": example["text"]} for example in lines
    ]
    processor_config = vLLMEngineProcessorConfig(
        model_source=model_name,
        engine_kwargs={
            "enable_lora": True,
            "max_lora_rank": lora_rank,
            "max_loras": 1,
            "dtype": "half",
            "max_model_len": 4096,
        },
        concurrency=1,
        batch_size=8,
        dynamic_lora_loading_path=adapter_path,
    )
    processor = build_llm_processor(
        processor_config,
        preprocess=lambda row: dict(
            messages=[
                {"role": "system", "content": "You are an expert speech evaluator."},
                {"role": "user", "content": row["prompt"]},
            ],
            sampling_params=dict(
                temperature=0.2,
                max_tokens=512,
            )
        ),
        postprocess=lambda row: dict(
            answer=row["generated_text"],
            text=row["text"],
        ),
    )
    ds = ray.data.from_items(prompts)
    ds = processor(ds)
    ds = ds.materialize()
    with open(output_path, "w") as fout:
        for out in ds.take_all():
            fout.write(json.dumps({"text": out["text"], "label": out["answer"]}, ensure_ascii=False) + "\n")
    ray.shutdown()

# ------------- Main Entrypoint -------------
if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    llm_config = config['llm_label']
    model_name = llm_config["model"]
    criteria = config["criteria"]
    input_file = llm_config['input_test_path']
    
    adapter = llm_config['adapter']
    use_ray = llm_config['use_ray']
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if adapter:
        import gc
        output_folder = llm_config['fine_tune']['output_folder']
        input_adapter_folder = llm_config['fine_tune']['input_adapter_folder']

        folders = [f for f in os.listdir(input_adapter_folder) if f.startswith("checkpoint") and os.path.isdir(os.path.join(input_adapter_folder, f))]
        for i, f in enumerate(folders, start=1):
            if use_ray:
                output_file = f'fine_tune_predictions_epoch{i+1}_ray.jsonl'
                label_batch_ray_lora(input_file, os.path.join(output_folder, output_file), criteria, model_name, tokenizer, os.path.join(input_adapter_folder, f))
            else:
                output_file = f'fine_tune_predictions_epoch{i+1}.jsonl'
                label_batch_local_lora(input_file, os.path.join(output_folder, output_file), criteria, model_name, os.path.join(input_adapter_folder, f), tokenizer)
            torch.cuda.empty_cache()
            gc.collect()
    else:
        output_folder = llm_config['pretrain']['output_folder']
        if use_ray:
            output_file = 'pretrain_predictions_ray.jsonl'
            label_batch_ray(input_file, os.path.join(output_folder, output_file), criteria, model_name, tokenizer)
        else:
            output_file = 'pretrain_predictions.jsonl'
            label_batch_local(input_file, os.path.join(output_folder, output_file), criteria, model_name, tokenizer)