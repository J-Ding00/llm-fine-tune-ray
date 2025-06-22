from datasets import load_dataset
import yaml
import json

def load_webtext_dataset(n_samples, output_file, dataset_name, split):
    """
    Streams n_samples from a Hugging Face dataset and writes to output_file as JSONL.
    """
    ds = load_dataset(dataset_name, split=split, streaming=True)
    with open(output_file, "w") as f:
        for i, item in enumerate(ds):
            item['id'] = i + 1
            f.write(json.dumps(item) + "\n")
            if (i + 1) >= n_samples:
                break
    print(f"Saved {n_samples} samples to {output_file}")

if __name__ == "__main__":
    config_path = "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    dp_cfg = config["load_data"]
    n_samples = dp_cfg["n_samples"]
    output_file = dp_cfg["raw_path"]
    dataset_name = dp_cfg["hf_streaming_name"]
    split = dp_cfg["hf_split"]
    load_webtext_dataset(n_samples, output_file, dataset_name, split)
