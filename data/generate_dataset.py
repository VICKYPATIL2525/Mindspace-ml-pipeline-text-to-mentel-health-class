import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from langchain_core.prompts import PromptTemplate
from llm_config import get_llms


# -------------------------------
# Prompt Template
# -------------------------------

conversation_prompt = PromptTemplate(
    input_variables=["language"],
    template="""
You are a clinical psychologist creating synthetic therapy conversations 
for research on mental health screening systems.

Generate a realistic therapy conversation between:

Therapist (Dr)
Patient (Client)

Language: {language}

Conversation requirements:

• Duration: 6–7 minutes of dialogue (~700–900 words)
• Natural emotional flow
• Therapist asks open-ended questions
• Patient answers with emotional detail
• Include pauses, uncertainty, and storytelling
• Simulate real therapy environment

Possible themes:

- stress
- anxiety
- sadness
- work pressure
- sleep issues
- relationships
- uncertainty about future

STRICT FORMAT:

Dr: question

Client: response

Dr: question

Client: response

Continue conversation naturally.

Do NOT summarize.

Produce a full transcript.
"""
)


# -------------------------------
# Generate Single Sample
# -------------------------------

def generate_single_sample(llm_model, model_name, language, sample_index):
    tqdm.write(f"  [→ START          ]  {model_name:<18} | {language:<8} | sample {sample_index}")
    prompt = conversation_prompt.format(language=language)
    response = llm_model.invoke(prompt)
    return {
        "model": model_name,
        "language": language,
        "sample_id": f"{model_name}_{language}_{sample_index}",
        "transcript": response.content
    }


# -------------------------------
# Dataset Generator (Parallel)
# -------------------------------

def generate_dataset():

    languages = ["English", "Hindi", "Marathi"]
    models = get_llms()
    n_samples = 3

    # Build all tasks
    tasks = [
        (model_name, llm_model, lang, i)
        for model_name, llm_model in models.items()
        for lang in languages
        for i in range(n_samples)
    ]

    total = len(tasks)
    done_count = 0
    failed_count = 0
    lock = threading.Lock()
    start_time = time.time()

    print(f"\n{'='*60}")
    print(f"  DATASET GENERATION STARTED")
    print(f"  Total tasks : {total}  (3 models × 3 languages × {n_samples} samples)")
    print(f"  Workers     : 6 parallel threads")
    print(f"{'='*60}\n")

    results = [None] * total

    with ThreadPoolExecutor(max_workers=6) as executor:
        future_to_idx = {
            executor.submit(generate_single_sample, llm_model, model_name, lang, i): idx
            for idx, (model_name, llm_model, lang, i) in enumerate(tasks)
        }

        with tqdm(total=total, desc="Progress", unit="sample",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                model_name, _, lang, i = tasks[idx]

                with lock:
                    try:
                        results[idx] = future.result(timeout=120)  # 2 min max per call
                        done_count += 1
                        elapsed = time.time() - start_time
                        avg_per_task = elapsed / done_count
                        remaining = (total - done_count) * avg_per_task
                        tqdm.write(
                            f"  [✓ DONE  {done_count:>2}/{total}]  "
                            f"{model_name:<18} | {lang:<8} | sample {i}  "
                            f"| remaining ~{remaining:.0f}s"
                        )
                    except Exception as e:
                        failed_count += 1
                        done_count += 1
                        tqdm.write(
                            f"  [✗ ERROR {done_count:>2}/{total}]  "
                            f"{model_name:<18} | {lang:<8} | sample {i}  "
                            f"| {e}"
                        )
                    pbar.update(1)

    elapsed_total = time.time() - start_time
    success = total - failed_count
    print(f"\n{'='*60}")
    print(f"  DONE  — {success}/{total} succeeded, {failed_count} failed")
    print(f"  Total time: {elapsed_total:.1f}s")
    print(f"{'='*60}\n")

    return [r for r in results if r is not None]


# -------------------------------
# Save Dataset
# -------------------------------

def save_dataset(dataset, filename="synthetic_therapy_dataset.json"):

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"\nDataset saved to {filename}")


# -------------------------------
# Main
# -------------------------------

if __name__ == "__main__":

    dataset = generate_dataset()

    save_dataset(dataset)