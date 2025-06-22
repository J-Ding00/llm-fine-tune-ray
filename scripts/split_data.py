import json
import os
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split

def get_score_bin(score, threshold):
    return "high" if score > threshold else "low"

def get_strat_key(sample, traits, threshold):
    try:
        label = json.loads(sample["label"])
        bins = [get_score_bin(label[t]["score"], threshold) for t in traits if t in label]
        return "_".join(bins)
    except Exception as e:
        print(e)
        return "unknown"

def write_jsonl(samples, path):
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

def split_data(input_path, traits, threshold, output_dir, seed=42, train_ratio=0.8, val_ratio=0.1):
    with open(input_path, "r") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    keys = [get_strat_key(s, traits, threshold) for s in samples]
    # print(keys)
    # from collections import Counter
    # print(Counter(keys))

    # 1st split: train vs temp (val+test)
    train_samples, temp_samples, train_keys, temp_keys = train_test_split(
        samples, keys, stratify=keys, test_size=1 - train_ratio, random_state=seed
    )

    # 2nd split: val vs test (from temp)
    relative_val_ratio = val_ratio / (1 - train_ratio)  # adjust for reduced size
    val_samples, test_samples, _, _ = train_test_split(
        temp_samples, temp_keys, stratify=temp_keys, test_size=1 - relative_val_ratio, random_state=seed
    )

    # Output files
    input_name = Path(input_path).stem
    os.makedirs(f"{output_dir}/train", exist_ok=True)
    os.makedirs(f"{output_dir}/val", exist_ok=True)
    os.makedirs(f"{output_dir}/test", exist_ok=True)

    write_jsonl(train_samples, f"{output_dir}/train/{input_name}_train.jsonl")
    write_jsonl(val_samples, f"{output_dir}/val/{input_name}_val.jsonl")
    write_jsonl(test_samples, f"{output_dir}/test/{input_name}_test.jsonl")

    print("Data split complete.")
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

if __name__ == "__main__":

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    sd_config = config['split_data']
    input_path = sd_config["input"]
    output_dir = sd_config["output_dir"]
    traits = sd_config["stratified_criteria"]
    threshold = sd_config["threshold"]
    train_ratio = sd_config["train_ratio"]
    val_ratio = sd_config["val_ratio"]

    split_data(
        input_path=input_path,
        traits=traits,
        threshold=threshold,
        output_dir=output_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio
    )