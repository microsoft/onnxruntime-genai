"""GSM8K evaluation: baseline vs speculative decoding.

Evaluates math reasoning quality on GSM8K test set (1319 problems).
Extracts the final numerical answer and compares against ground truth.
"""

import ctypes
import json
import os
import re
import sys
import time

ctypes.CDLL(r"C:\Project\speculative\onnxruntime-genai\build\Windows\Release\Release\onnxruntime.dll")
import onnxruntime_genai as og


GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"

# 5-shot CoT examples from GSM8K train set (standard evaluation format)
FEW_SHOT_EXAMPLES = """Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Answer: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. #### 6

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Answer: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. #### 5

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Answer: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. #### 39

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Answer: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. #### 8

Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
Answer: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 2 + 2 = 4 more toys. 5 + 4 = 9. #### 9

"""


def load_gsm8k(max_problems=None):
    """Load GSM8K test set, downloading if needed."""
    cache_dir = os.path.join(os.path.dirname(__file__), ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "gsm8k_test.jsonl")

    if not os.path.exists(cache_path):
        print(f"Downloading GSM8K test set...")
        import urllib.request
        urllib.request.urlretrieve(GSM8K_URL, cache_path)

    data = []
    with open(cache_path, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    if max_problems:
        data = data[:max_problems]
    return data


def extract_answer(text):
    """Extract the final numerical answer from model output.
    GSM8K answers are formatted as '#### <number>' in ground truth.
    Models may output the number in various formats."""
    # Look for #### pattern (GSM8K format)
    match = re.search(r'####\s*([+-]?[\d,]+\.?\d*)', text)
    if match:
        return match.group(1).replace(',', '')

    # Look for "the answer is <number>" pattern
    match = re.search(r'(?:the answer is|answer:|= )\s*\$?([+-]?[\d,]+\.?\d*)', text, re.IGNORECASE)
    if match:
        return match.group(1).replace(',', '')

    # Fall back to last number in text
    numbers = re.findall(r'[+-]?\d[\d,]*\.?\d*', text)
    if numbers:
        return numbers[-1].replace(',', '')

    return None


def extract_ground_truth(answer_text):
    """Extract ground truth number from GSM8K answer field."""
    match = re.search(r'####\s*([+-]?[\d,]+\.?\d*)', answer_text)
    if match:
        return match.group(1).replace(',', '')
    return None


def generate(model_path, prompt, max_new_tokens=512, speculative=False):
    """Generate a completion."""
    m = og.Model(model_path)
    t = og.Tokenizer(m)
    tokens = t.encode(prompt)

    p = og.GeneratorParams(m)
    opts = {"max_length": len(tokens) + max_new_tokens, "do_sample": False, "top_k": 1}
    if speculative:
        opts["speculative_ngram_size"] = 3
        opts["speculative_max_draft_tokens"] = 5
    p.set_search_options(**opts)

    g = og.Generator(m, p)
    g.append_tokens(tokens)

    start = time.perf_counter()
    while not g.is_done():
        g.generate_next_token()
    elapsed = time.perf_counter() - start

    full_text = t.decode(g.get_sequence(0))
    completion = full_text[len(prompt):]

    # Stop at next question or double newline to avoid the model continuing
    for stop in ["\nQuestion:", "\n\nQuestion", "\n\n\n"]:
        idx = completion.find(stop)
        if idx != -1:
            completion = completion[:idx]

    return completion, elapsed


def run_gsm8k(model_path, speculative=False, max_problems=None):
    """Run GSM8K evaluation."""
    mode = "speculative" if speculative else "baseline"

    print(f"\nLoading GSM8K test set...")
    dataset = load_gsm8k(max_problems)

    print(f"\n{'='*60}")
    print(f"GSM8K Evaluation ({mode})")
    print(f"Model: {model_path}")
    print(f"Problems: {len(dataset)}")
    print(f"{'='*60}")

    correct = 0
    total = 0
    total_time = 0
    results = []

    for i, example in enumerate(dataset):
        question = example["question"]
        ground_truth = extract_ground_truth(example["answer"])

        prompt = f"{FEW_SHOT_EXAMPLES}Question: {question}\nAnswer:"
        completion, elapsed = generate(
            model_path, prompt, max_new_tokens=512, speculative=speculative
        )
        total_time += elapsed

        predicted = extract_answer(completion)
        is_correct = (predicted == ground_truth) if predicted and ground_truth else False
        if is_correct:
            correct += 1
        total += 1

        results.append({
            "task_id": i,
            "ground_truth": ground_truth,
            "predicted": predicted,
            "correct": is_correct,
            "time": elapsed,
        })

        if (i + 1) % 50 == 0 or (i + 1) == len(dataset):
            acc = correct / total * 100
            print(f"  [{mode}] {i+1}/{len(dataset)} done | acc={acc:.1f}% ({correct}/{total}) | {total_time:.0f}s", flush=True)

    accuracy = correct / total if total > 0 else 0.0
    print(f"\n  {mode} Results:")
    print(f"    Accuracy:   {accuracy:.4f} ({accuracy*100:.1f}%) [{correct}/{total}]")
    print(f"    Total time: {total_time:.1f}s")
    print(f"    Avg time:   {total_time/total:.2f}s/problem")

    return {
        "mode": mode,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "total_time": total_time,
        "results": results,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GSM8K evaluation for speculative decoding")
    parser.add_argument("-m", "--model", default=r"C:\Project\models\qwen3-0.6b\cpu_and_mobile\v4")
    parser.add_argument("-n", "--num_problems", type=int, default=200,
                        help="Number of problems (default 200, use 1319 for full)")
    args = parser.parse_args()

    baseline = run_gsm8k(args.model, speculative=False, max_problems=args.num_problems)
    speculative = run_gsm8k(args.model, speculative=True, max_problems=args.num_problems)

    # Per-problem comparison
    both_correct = 0
    base_only = 0
    spec_only = 0
    both_wrong = 0
    for b, s in zip(baseline["results"], speculative["results"]):
        if b["correct"] and s["correct"]:
            both_correct += 1
        elif b["correct"] and not s["correct"]:
            base_only += 1
        elif not b["correct"] and s["correct"]:
            spec_only += 1
        else:
            both_wrong += 1

    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Metric':<20} {'Baseline':>12} {'Speculative':>12} {'Delta':>12}")
    print(f"  {'-'*56}")
    print(f"  {'Accuracy':<20} {baseline['accuracy']:>11.1%} {speculative['accuracy']:>11.1%} {speculative['accuracy']-baseline['accuracy']:>+11.1%}")
    print(f"  {'Correct':<20} {baseline['correct']:>12} {speculative['correct']:>12} {speculative['correct']-baseline['correct']:>+12}")
    print(f"  {'Total time':<20} {baseline['total_time']:>11.1f}s {speculative['total_time']:>11.1f}s {baseline['total_time']/speculative['total_time']:>11.2f}x")
    print(f"  {'Avg time/problem':<20} {baseline['total_time']/baseline['total']:>11.2f}s {speculative['total_time']/speculative['total']:>11.2f}s")

    print(f"\n  Per-problem breakdown:")
    print(f"    Both correct:    {both_correct:>4} ({both_correct/baseline['total']*100:.1f}%)")
    print(f"    Baseline only:   {base_only:>4} ({base_only/baseline['total']*100:.1f}%)")
    print(f"    Speculative only:{spec_only:>4} ({spec_only/baseline['total']*100:.1f}%)")
    print(f"    Both wrong:      {both_wrong:>4} ({both_wrong/baseline['total']*100:.1f}%)")

    delta = abs(speculative['accuracy'] - baseline['accuracy'])
    if delta < 0.03:
        print(f"\n  ✓ No quality regression (delta={delta:.1%} < 3% threshold)")
    else:
        direction = "improvement" if speculative['accuracy'] > baseline['accuracy'] else "regression"
        print(f"\n  ⚠ Quality {direction} detected (delta={delta:.1%})")
