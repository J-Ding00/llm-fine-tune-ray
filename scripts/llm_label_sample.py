import json
import yaml
import os
from tqdm import tqdm
# from utils.utils import ray_data_step_logger
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import ray

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

@ray.remote(num_gpus=1)
def label_batch_local(input_path, criteria, model_name, tokenizer):
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    with open(input_path, "r") as f:
        lines = [json.loads(l) for l in f if l.strip()]
    criteria_format = {c: {"score": "<int>", "feedback": "<string>"} for c in criteria}
    results = []
    for example in tqdm(lines, desc="Labeling samples (local)"):
        transcript = example["text"]
        try:
            feedback = generate_local_feedback(model, tokenizer, transcript, criteria, criteria_format)
            labeled = {"text": transcript, "label": feedback, "id":example["id"]}
            results.append(labeled)
        except Exception:
            print("[Error] Skipping sample")
    return results

@ray.remote(num_gpus=1)
def label_batch_local_lora(input_path, criteria, model_name, adapter_path, tokenizer):
    from peft import PeftModel
    base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto",)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    with open(input_path, "r") as f:
        lines = [json.loads(l) for l in f if l.strip()]
    criteria_format = {c: {"score": "<int>", "feedback": "<string>"} for c in criteria}
    results = []
    for example in tqdm(lines, desc="Labeling samples (local + LoRA)"):
        transcript = example["text"]
        try:
            feedback = generate_local_feedback(model, tokenizer, transcript, criteria, criteria_format)
            labeled = {"text": transcript, "label": feedback, "id":example["id"]}
            # fout.write(json.dumps(labeled, ensure_ascii=False) + "\n")
            results.append(labeled)
        except Exception:
            print("[Error] Skipping sample")
    return results

def label_batch_ray(input_path, output_path, criteria, model_name):
    import ray
    from ray.data.llm import build_llm_processor, vLLMEngineProcessorConfig
    criteria_format = {c: {"score": "<int>", "feedback": "<string>"} for c in criteria}
    with open(input_path, "r") as f:
        lines = [json.loads(l) for l in f if l.strip()]

    prompts = [{"text": example["text"], "id": example["id"]} for example in lines]

    def preprocess(row):
        return dict(
            messages=build_prompt(row['text'], criteria, criteria_format),
            sampling_params=dict(
                temperature=0.2,
                max_tokens=512,
            )
        )

    processor_config = vLLMEngineProcessorConfig(
        model_source=model_name,
        engine_kwargs={
            "enable_chunked_prefill": True,
            "max_num_batched_tokens": 16384, #8192
        },
        concurrency=1,
        batch_size=8,
    )
    processor = build_llm_processor(
        processor_config,
        preprocess=preprocess,
        postprocess=lambda row: dict(
            answer=row["generated_text"],
            text=row["text"],
            id=row['id'],
        ),
    )
    ds = ray.data.from_items(prompts)
    ds = processor(ds)
    ds = ds.materialize()
    with open(output_path, "w") as fout:
        for out in ds.take_all():
            fout.write(json.dumps({"text": out["text"], "label": out["answer"], "id":out['id']}) + "\n")
    ray.shutdown()

def label_batch_ray_lora(input_path, output_path, criteria, model_name, tokenizer, adapter_path, lora_rank=16):
    import ray
    from ray.data.llm import build_llm_processor, vLLMEngineProcessorConfig
    criteria_format = {c: {"score": "<int>", "feedback": "<string>"} for c in criteria}
    with open(input_path, "r") as f:
        lines = [json.loads(l) for l in f if l.strip()]
    # prompts = [
    #     {"prompt": tokenizer.apply_chat_template(
    #         build_prompt(example["text"], criteria, criteria_format),
    #         tokenize=False,
    #         add_generation_prompt=True
    #     ), "text": example["text"]} for example in lines
    # ]
    prompts = [{"text": example["text"], "id": example["id"]} for example in lines]
    def preprocess(row):
        return dict(
            # model='VinitT/Commentary-qwen-3B',
            messages=build_prompt(row['text'], criteria, criteria_format),
            sampling_params=dict(
                temperature=0.2,
                max_tokens=512,
                lora_adapter_name=adapter_path
                # detokenize=False
            )
        )
    processor_config = vLLMEngineProcessorConfig(
        model_source=model_name,
        engine_kwargs={
            "enable_lora": True,
            "max_lora_rank": 16,
            "max_loras": 1,
            "enable_chunked_prefill": True,
            # "max_num_batched_tokens": 16384,
            'dtype': torch.bfloat16,
            "max_model_len": 2048,
        },
        concurrency=1,
        batch_size=1,
        dynamic_lora_loading_path=adapter_path,
    )
    processor = build_llm_processor(
        processor_config,
        preprocess=preprocess,
        postprocess=lambda row: dict(
            answer=row["generated_text"],
            text=row["text"],
            id=row['id'],
        ),
    )
    ds = ray.data.from_items(prompts)
    ds = processor(ds)
    ds = ds.materialize()
    with open(output_path, "w") as fout:
        for out in ds.take_all():
            fout.write(json.dumps({"text": out["text"], "label": out["answer"], "id":out['id']}) + "\n")
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
        # os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"
        # ray.init(runtime_env={"working_dir": "."})
        import gc
        output_folder = llm_config['fine_tune']['output_folder']
        input_adapter_folder = llm_config['fine_tune']['input_adapter_folder']

        folders = [f for f in os.listdir(input_adapter_folder) if f.startswith("checkpoint") and os.path.isdir(os.path.join(input_adapter_folder, f))]
        for i, f in enumerate(folders, start=1):
            if use_ray:
                output_file = f'fine_tune_predictions_epoch{i+2}_ray.jsonl'
                label_batch_ray_lora(input_file, os.path.join(output_folder, output_file), criteria, model_name, tokenizer, os.path.join(input_adapter_folder, f))
            else:
                ray.init(runtime_env={"working_dir": "/home/ray/default/llm-fine-tune-ray"})
                output_file = f'fine_tune_predictions_epoch{i+2}.jsonl'
                result = ray.get(label_batch_local_lora.remote(input_file, criteria, model_name, os.path.join(input_adapter_folder, f), tokenizer))
                with open(os.path.join(output_folder, output_file), "w") as fout:
                    for item in result:
                        fout.write(json.dumps(item) + "\n")
            # torch.cuda.empty_cache()
            # gc.collect()
    else:
        output_folder = llm_config['pretrain']['output_folder']
        if use_ray:
            output_file = 'pretrain_predictions_ray.jsonl'
            label_batch_ray(input_file, os.path.join(output_folder, output_file), criteria, model_name)
        else:
            ray.init(runtime_env={"working_dir": "/home/ray/default/llm-fine-tune-ray"})
            output_file = 'pretrain_predictions.jsonl'
            # label_batch_local(input_file, os.path.join(output_folder, output_file), criteria, model_name, tokenizer)
            result = ray.get(label_batch_local.remote(input_file, criteria, model_name, tokenizer))
            with open(os.path.join(output_folder, output_file), "w") as fout:
                for item in result:
                    fout.write(json.dumps(item) + "\n")