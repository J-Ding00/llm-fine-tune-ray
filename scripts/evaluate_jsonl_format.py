import json
import yaml
import re
from sklearn.metrics import mean_squared_error
from collections import defaultdict
from bert_score import score as bert_score
import torch
import os

def compare_multiple_score_mse(generated_files, ground_truth_file, traits):
    """
    Compare score MSE between multiple model output files and ground truth label file.
    Only lines that are valid across all generated files are included.

    Args:
        generated_files (List[str]): List of paths to model output .jsonl files.
        ground_truth_file (str): Path to clean ground truth .jsonl file.
        traits (List[str]): List of trait names to compare scores for.

    Returns:
        dict: {generated_file: {trait: MSE, ..., "overall_avg_mse": avg MSE}} for each generated file
        str: Summary message
    """
    # One trait_scores per generated file
    all_trait_scores = {gen_file: defaultdict(lambda: {'pred': [], 'gt': []}) for gen_file in generated_files}

    total_lines = 0
    valid_lines = 0

    # Open all generated files and ground truth file
    file_handlers = [open(f, 'r') for f in generated_files]
    gt_handler = open(ground_truth_file, 'r')

    try:
        for lines in zip(*file_handlers, gt_handler):
            total_lines += 1
            gen_lines = lines[:-1]
            gt_line = lines[-1]

            try:
                gt = json.loads(gt_line)
                gt_label = json.loads(gt['label'])

                pred_labels = []
                all_valid = True
                
                for gen_line in gen_lines:
                    gen = json.loads(gen_line)
                    pred_label = json.loads(extract_and_fix_json(gen['label']))
                    valid, _ = is_valid_json_structure(pred_label, traits)
                    if not valid:
                        all_valid = False
                        break
                    pred_labels.append(pred_label)

                if all_valid:
                    for gen_file, pred_label in zip(generated_files, pred_labels):
                        for trait in traits:
                            pred_score = int(pred_label[trait]["score"])
                            gt_score = int(gt_label[trait]["score"])
                            all_trait_scores[gen_file][trait]["pred"].append(pred_score)
                            all_trait_scores[gen_file][trait]["gt"].append(gt_score)
                    valid_lines += 1

            except Exception as e:
                continue

    finally:
        for f in file_handlers:
            f.close()
        gt_handler.close()
    results = {}
    for gen_file in generated_files:
        trait_results = {}
        for trait in traits:
            pred = all_trait_scores[gen_file][trait]["pred"]
            gt = all_trait_scores[gen_file][trait]["gt"]
            if pred and gt:
                trait_results[trait] = mean_squared_error(gt, pred)
        if trait_results:
            trait_results["overall_avg_mse"] = sum(trait_results.values()) / len(trait_results)
        else:
            trait_results["overall_avg_mse"] = None
        results[gen_file] = trait_results
    # Pretty print summary
    msg = "\nMSE Metrics across generated files:\n\n"
    for gen_file, trait_mses in results.items():
        msg += f"Generated File: {gen_file}\n"
        for trait, mse in trait_mses.items():
            msg += f"  {trait}: {mse:.4f}\n"
        msg += "\n"

    msg += f"Compared {valid_lines}/{total_lines} lines successfully (only counting lines valid in all files).\n"
    return results, msg

def compare_multiple_score_mse_bert(generated_files, ground_truth_file, traits):
    """
    Compare score MSE and BERTScore Precision/Recall/F1 between multiple model output files and ground truth label file.
    Only lines that are valid across all generated files are included.

    Returns:
        dict: {generated_file: {trait: MSE + BERTScore metrics, ..., "overall_avg_mse": ..., "overall_avg_bertscore": ...}}
        str: Summary message
    """
    all_trait_scores = {
        gen_file: defaultdict(lambda: {'pred': [], 'gt': [], 'pred_fb': [], 'gt_fb': []})
        for gen_file in generated_files
    }

    total_lines = 0
    valid_lines = 0

    file_handlers = [open(f, 'r') for f in generated_files]
    gt_handler = open(ground_truth_file, 'r')

    try:
        for lines in zip(*file_handlers, gt_handler):
            total_lines += 1
            gen_lines = lines[:-1]
            gt_line = lines[-1]

            try:
                gt = json.loads(gt_line)
                gt_label = json.loads(gt['label'])

                pred_labels = []
                all_valid = True

                for gen_line in gen_lines:
                    gen = json.loads(gen_line)
                    pred_label = json.loads(extract_and_fix_json(gen['label']))
                    valid, _ = is_valid_json_structure(pred_label, traits)
                    if not valid:
                        all_valid = False
                        break
                    pred_labels.append(pred_label)

                if all_valid:
                    for gen_file, pred_label in zip(generated_files, pred_labels):
                        for trait in traits:
                            pred_score = int(pred_label[trait]["score"])
                            gt_score = int(gt_label[trait]["score"])
                            pred_fb = pred_label[trait]["feedback"]
                            gt_fb = gt_label[trait]["feedback"]

                            all_trait_scores[gen_file][trait]["pred"].append(pred_score)
                            all_trait_scores[gen_file][trait]["gt"].append(gt_score)
                            all_trait_scores[gen_file][trait]["pred_fb"].append(pred_fb)
                            all_trait_scores[gen_file][trait]["gt_fb"].append(gt_fb)
                    valid_lines += 1

            except Exception:
                continue
    finally:
        for f in file_handlers:
            f.close()
        gt_handler.close()

    results = {}
    for gen_file in generated_files:
        trait_results = {}
        trait_bertscores = []

        for trait in traits:
            pred = all_trait_scores[gen_file][trait]["pred"]
            gt = all_trait_scores[gen_file][trait]["gt"]
            pred_fb = all_trait_scores[gen_file][trait]["pred_fb"]
            gt_fb = all_trait_scores[gen_file][trait]["gt_fb"]

            # MSE
            if pred and gt:
                trait_results[trait] = mean_squared_error(gt, pred)

            # BERTScore P/R/F1
            if pred_fb and gt_fb:
                P, R, F1 = bert_score(
                    pred_fb,
                    gt_fb,
                    lang="en",
                    verbose=False,
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
                trait_results[f"{trait}_bertscore_precision"] = P.mean().item()
                trait_results[f"{trait}_bertscore_recall"] = R.mean().item()
                trait_results[f"{trait}_bertscore_f1"] = F1.mean().item()
                trait_bertscores.append(F1.mean().item())

        # Overall MSE and BERTScore-F1
        trait_results["overall_avg_mse"] = (
            sum(v for k, v in trait_results.items() if not k.endswith("bertscore_precision")
                                                        and not k.endswith("bertscore_recall")
                                                        and not k.endswith("bertscore_f1")
                                                        and not k.startswith("overall"))
            / len(traits)
        )
        trait_results["overall_avg_bertscore_f1"] = (
            sum(trait_bertscores) / len(trait_bertscores) if trait_bertscores else None
        )
        results[gen_file] = trait_results

    # Pretty print summary
    msg = "\nMSE and BERTScore Metrics across generated files:\n\n"
    for gen_file, metrics in results.items():
        msg += f"Generated File: {gen_file}\n"
        for trait, val in metrics.items():
            if val is not None:
                msg += f"  {trait}: {val:.4f}\n"
        msg += "\n"
    msg += f"Compared {valid_lines}/{total_lines} lines successfully (only counting lines valid in all files).\n"

    return results, msg

def extract_and_fix_json(text):
    """
    Extracts and fixes a JSON block from a text string.
    - Tries to extract JSON from markdown-style code blocks or brace matching.
    - Applies simple fixes for trailing commas and missing closing braces.
    """
    # Extract from fenced code block
    match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if not match:
        # Fallback: extract first curly-braced block
        match = re.search(r'(\{.*\})', text, re.DOTALL)
    json_text = match.group(1).strip() if match else text.strip()

    # Fix trailing commas before closing } or ]
    json_text = re.sub(r',\s*([}\]])', r'\1', json_text)

    # Naively add closing brace if there's a mismatch
    if json_text.count('{') > json_text.count('}'):
        json_text += '}'

    return json_text

def is_valid_json_structure(obj, criteria):
    for field in criteria:
        if field not in obj:
            return False, f"Missing field: {field}"
        trait = obj[field]
        if not isinstance(trait, dict):
            return False, f"Field '{field}' is not a dict"
        if "score" not in trait or "feedback" not in trait:
            return False, f"Field '{field}' missing 'score' or 'feedback'"
        try:   
            if int(trait["score"]) < 1 or int(trait["score"]) > 10:
                return False, f"Score in '{field}' outside range"
        except Exception as e:
            return False, f"Score in '{field}' is not int"
        if not isinstance(trait["feedback"], str):
            return False, f"Feedback in '{field}' is not str"
    return True, None

def validate_and_analyze_jsonl(path, criteria, score_threshold):
    with open(path, "r") as f:
        lines = f.readlines()

    total = len(lines)
    valid = 0
    invalid_lines = []
    trait_stats = {trait: {
        f"count_above_{score_threshold}": 0,
        f"count_{score_threshold}_or_below": 0,
        "sum": 0,
        "scores": [],
        "total": 0
    } for trait in criteria}

    for idx, line in enumerate(lines):
        try:
            output = json.loads(line)['label']
        except Exception as e:
            invalid_lines.append((idx, f"Outer JSON error: {e}"))
            continue

        try:
            label = json.loads(output)
        except json.JSONDecodeError:
            try:
                fixed = extract_and_fix_json(output)
                label = json.loads(fixed)
            except Exception as e2:
                invalid_lines.append((idx, f"Unrecoverable JSON decode error: {e2}"))
                continue

        ok, msg = is_valid_json_structure(label, criteria)
        if not ok:
            invalid_lines.append((idx, msg))
            continue

        valid += 1
        for trait in criteria:
            score = int(label[trait]["score"])
            trait_stats[trait]["scores"].append(score)
            trait_stats[trait]["sum"] += score
            trait_stats[trait]["total"] += 1
            if score > score_threshold:
                trait_stats[trait][f"count_above_{score_threshold}"] += 1
            else:
                trait_stats[trait][f"count_{score_threshold}_or_below"] += 1

    metrics = [
        f"\n{path} check with score threshold {score_threshold}:\n",
        f"Valid samples: {valid}/{total}\n",
        f"Invalid samples: {total - valid}\n",
        "\nSample errors (up to 10 shown):\n"
    ]
    for idx, err in invalid_lines[:10]:
        metrics.append(f"  Line {idx + 1}: {err}\n")

    metrics.append("\nScoring summary:\n")
    for trait, stats in trait_stats.items():
        total = stats["total"]
        if total == 0: continue
        avg = stats["sum"] / total
        metrics.append(f"\nTrait: {trait}\n")
        metrics.append(f"  Avg Score: {avg:.2f}\n")
        metrics.append(f"  >{score_threshold}: {stats[f'count_above_{score_threshold}']} | <={score_threshold}: {stats[f'count_{score_threshold}_or_below']}\n")
        metrics.append(f"  Min: {min(stats['scores'])}, Max: {max(stats['scores'])}\n")
    metrics.append('\n\n')

    return metrics

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    eval_file_list = config['evaluation']['jsonl_format_in']
    # eval_file_list = [os.path.join(config['evaluation']['jsonl_generated_data_in'], f) for f in os.listdir(config['evaluation']['jsonl_generated_data_in'])]
    output_path = config['evaluation']['jsonl_format_out']
    criteria = config['criteria']
    score_threshold = config['evaluation']['score_threshold']
    mse = config['evaluation']['mse']

    with open(output_path, "a") as out:
        for file in eval_file_list:
            if file.endswith('.jsonl'):
                for score in score_threshold:
                    metrics = validate_and_analyze_jsonl(file, criteria, score)
                    out.writelines(metrics)
                    print(f"Finished: {file}")

        if mse and len(eval_file_list) > 1:
            mse_scores, msg = compare_multiple_score_mse(
                generated_files=eval_file_list[:-1],
                ground_truth_file=eval_file_list[-1],
                traits=criteria
            )
            out.write(msg)

    print("All files processed. Metrics written to:", output_path)
