# import os
# import json
# import yaml
# from collections import defaultdict
# from pathlib import Path

# def summarize_split_stats(config_path):
#     with open(config_path, "r") as f:
#         config = yaml.safe_load(f)

#     traits = config["data_split"]["stratified_criteria"]
#     threshold = config["data_split"]["threshold"]
#     prefix = Path(config["data_split"]["input"]).stem
#     base_dir = Path(config["data_split"]["output_dir"])

#     def get_score_bin(score):
#         return "high" if score > threshold else "low"

#     for split in ["train", "val", "test"]:
#         path = base_dir / split / f"{prefix}_{split}.jsonl"
#         with open(path, "r") as f:
#             lines = [json.loads(line.strip()) for line in f if line.strip()]

#         trait_counts = {trait: defaultdict(int) for trait in traits}

#         for sample in lines:
#             try:
#                 label = json.loads(sample["label"])
#                 for trait in traits:
#                     if trait in label:
#                         bin_key = get_score_bin(label[trait]["score"])
#                         trait_counts[trait][bin_key] += 1
#             except Exception as e:
#                 print(f"Skipping malformed sample: {e}")

#         print(f"\nStats for {split.capitalize()}: {len(lines)} samples")
#         for trait in traits:
#             total = sum(trait_counts[trait].values())
#             low = trait_counts[trait]["low"]
#             high = trait_counts[trait]["high"]
#             print(f"  Trait: {trait.ljust(18)} | low: {low:<4} ({low/total:.1%}) | high: {high:<4} ({high/total:.1%})")
    

# def average_transcript_length(jsonl_path):
#     total_words = 0
#     total_samples = 0

#     with open(jsonl_path, 'r') as f:
#         for line in f:
#             try:
#                 obj = json.loads(line)
#                 text = obj.get("text", "")
#                 word_count = len(text.strip().split())
#                 total_words += word_count
#                 total_samples += 1
#             except Exception as e:
#                 print(f"Skipping line due to error: {e}")

#     if total_samples == 0:
#         return 0

#     avg_length = total_words / total_samples
#     print(f"Average transcript length: {avg_length:.2f} words ({total_samples} samples)")
#     return avg_length

# if __name__ == "__main__":
#     # print_folder_structure(".")
#     # average_transcript_length('data/raw/openwebtext_sample.jsonl')
#     summarize_split_stats("config.yaml")
#     exit()

import time
import psutil
import os

def ray_data_step_logger(ds, step_name):
    stats = ds.stats()
    result = {
        "step": step_name,
        "num_rows": ds.count(),
        "ray_stats": stats,
    }
    print(f"[Ray: {step_name}] {result}")
    return result

class StepLogger:
    """For local single-process steps."""
    def __init__(self, name):
        self.name = name
        self.pid = os.getpid()

    def __enter__(self):
        self.start_time = time.perf_counter()
        self.start_mem = psutil.Process(self.pid).memory_info().rss / 1024**2  # MB
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.perf_counter()
        end_mem = psutil.Process(self.pid).memory_info().rss / 1024**2  # MB
        print(f"[{self.name}] took {end_time - self.start_time:.2f}s, "
              f"Memory: {self.start_mem:.2f}MB -> {end_mem:.2f}MB")



# For Ray remote functions
def ray_step_logger(fn):
    import ray

    @ray.remote
    def wrapper(*args, **kwargs):
        import time
        import psutil
        import os

        pid = os.getpid()
        start_time = time.perf_counter()
        start_mem = psutil.Process(pid).memory_info().rss / 1024**2  # MB

        result = fn(*args, **kwargs)

        end_time = time.perf_counter()
        end_mem = psutil.Process(pid).memory_info().rss / 1024**2  # MB
        log = {
            "step": fn.__name__,
            "time_sec": round(end_time - start_time, 2),
            "memory_mb_start": round(start_mem, 2),
            "memory_mb_end": round(end_mem, 2),
        }
        return result, log

    return wrapper