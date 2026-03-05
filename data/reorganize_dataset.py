import json
from collections import defaultdict

INPUT_FILE  = "synthetic_therapy_dataset.json"
OUTPUT_FILE = "dataset_by_model.json"

# Load flat dataset
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    dataset = json.load(f)

# Group by model → language → samples
grouped = defaultdict(lambda: defaultdict(list))
for sample in dataset:
    grouped[sample["model"]][sample["language"]].append(sample)

# Build ordered output: model → language → numbered samples
output = {}
for model_name, languages in grouped.items():
    output[model_name] = {}
    for language, samples in languages.items():
        output[model_name][language] = {
            f"sample_{i+1}": sample["transcript"]
            for i, sample in enumerate(samples)
        }

# Save
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

# Summary
print(f"\nSaved to: {OUTPUT_FILE}")
print(f"{'─'*50}")
for model_name, languages in output.items():
    print(f"\n  [{model_name}]")
    for lang, samples in languages.items():
        print(f"    {lang:<10} → {len(samples)} samples")
print(f"\n{'─'*50}")
print(f"  Total: {len(dataset)} samples\n")
