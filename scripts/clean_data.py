import json
import re
import yaml
import os

def json_simple_fix(text):
    text = re.sub(r',\s*([}\]])', r'\1', text)
    if text.count('{') > text.count('}'): text += '}'
    return text

def is_valid_json_structure(obj, criteria):
    for field in criteria:
        if field not in obj:
            return False
        trait = obj[field]
        if not isinstance(trait, dict):
            return False
        if "score" not in trait or "feedback" not in trait:
            return False
        if not isinstance(trait["score"], int):
            return False
        if trait["score"] < 1 or trait["score"] > 10:
            return False
        if not isinstance(trait["feedback"], str):
            return False
    return True

def filter_valid_examples(path, criteria, output_path):
    valid_lines = 0
    with open(path, "r") as fin, open(output_path, "a") as fout:
        for idx, line in enumerate(fin):
            try:
                outer = json.loads(line)
                del(outer['id'])
                raw_label = outer.get("label")

                # Try parsing the label, and fix if necessary
                try:
                    label = json.loads(raw_label)
                except json.JSONDecodeError:
                    fixed_label_str = json_simple_fix(raw_label)
                    label = json.loads(fixed_label_str)
                    outer["label"] = fixed_label_str  # Update with cleaned label

                # Validate structure
                if is_valid_json_structure(label, criteria):
                    valid_lines += 1
                    fout.write(json.dumps(outer) + "\n")

            except Exception:
                continue

    print(f"Filtered and saved {valid_lines} valid examples to {output_path}")

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # choose one dataset folder (original_webtext/augmented_webtext/combined_webtext) for data_cleaning
    # cd_config = config['clean_data']['webtext']
    # cd_config = config['clean_data']['generated']
    cd_config = config['clean_data']['combined']

    input_folder_list = cd_config['input_folder_list']
    output_file = cd_config['output_path']

    criteria = config['criteria']

    for input_folder in input_folder_list:
        for input_path in os.listdir(input_folder):
            if input_path.endswith('.jsonl'):
                filter_valid_examples(os.path.join(input_folder, input_path), criteria, output_file)
