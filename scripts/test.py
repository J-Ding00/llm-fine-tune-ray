import json
with open('predictions/fine_tune/fine_tune_predictions_epoch1_ray.jsonl', "r") as fin:
    with open('predictions/fine_tune/sfine_tune_predictions_epoch1_ray.jsonl', "w") as fout:
        out = sorted([json.loads(r) for r in fin], key=lambda x: x['id'])
        for item in out:
            fout.write(json.dumps(item) + "\n")