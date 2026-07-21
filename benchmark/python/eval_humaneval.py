"""HumanEval evaluation: baseline vs speculative decoding.

Generates completions for all 164 HumanEval problems with both modes,
evaluates pass@1, and reports whether speculative decoding causes
any quality regression.
"""

import ctypes
import json
import os
import sys
import time
import tempfile

ctypes.CDLL(r"C:\Project\speculative\onnxruntime-genai\build\Windows\Release\Release\onnxruntime.dll")
import onnxruntime_genai as og
from human_eval.data import write_jsonl, read_problems
from human_eval.evaluation import evaluate_functional_correctness


def generate_completion(model_path, prompt, max_new_tokens=256, speculative=False):
    """Generate a single completion."""
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

    seq = g.get_sequence(0)
    full_text = t.decode(seq)

    # Extract only the completion (remove the prompt)
    completion = full_text[len(prompt):]

    # Stop at common end markers for code generation
    for stop in ["\nclass ", "\ndef ", "\n#", "\nif __name__", "\nprint(", "\n```"]:
        idx = completion.find(stop)
        if idx != -1:
            completion = completion[:idx]

    return completion, elapsed


def run_humaneval(model_path, speculative=False, max_problems=None):
    """Run HumanEval on all problems."""
    mode = "speculative" if speculative else "baseline"
    problems = read_problems()

    if max_problems:
        problem_ids = list(problems.keys())[:max_problems]
        problems = {k: problems[k] for k in problem_ids}

    print(f"\n{'='*60}")
    print(f"Running HumanEval ({mode}) on {len(problems)} problems")
    print(f"Model: {model_path}")
    print(f"{'='*60}")

    samples = []
    total_time = 0
    for i, (task_id, problem) in enumerate(problems.items()):
        prompt = problem["prompt"]
        completion, elapsed = generate_completion(
            model_path, prompt, max_new_tokens=512, speculative=speculative
        )
        total_time += elapsed
        samples.append({"task_id": task_id, "completion": completion})

        if (i + 1) % 10 == 0 or (i + 1) == len(problems):
            print(f"  [{mode}] {i+1}/{len(problems)} done ({total_time:.1f}s total)", flush=True)

    # Write samples to temp file
    tmpdir = tempfile.mkdtemp()
    sample_file = os.path.join(tmpdir, f"samples_{mode}.jsonl")
    write_jsonl(sample_file, samples)

    # Evaluate — need all 164 problems for evaluate_functional_correctness,
    # so if partial, fill missing with empty completions
    all_problems = read_problems()
    attempted_ids = {s["task_id"] for s in samples}
    full_samples = list(samples)
    for task_id in all_problems:
        if task_id not in attempted_ids:
            full_samples.append({"task_id": task_id, "completion": ""})

    full_sample_file = os.path.join(tmpdir, f"samples_{mode}_full.jsonl")
    write_jsonl(full_sample_file, full_samples)

    print(f"\n  Evaluating {mode}...")
    results = evaluate_functional_correctness(full_sample_file, k=[1])
    # Recompute pass@1 only for the problems we actually attempted
    result_file = full_sample_file + "_results.jsonl"
    passed = 0
    total = 0
    with open(result_file) as f:
        for line in f:
            r = json.loads(line)
            if r["task_id"] in attempted_ids:
                total += 1
                if r["passed"]:
                    passed += 1
    pass_at_1 = passed / total if total > 0 else 0.0

    print(f"\n  {mode} Results:")
    print(f"    pass@1:     {pass_at_1:.4f} ({pass_at_1*100:.1f}%)")
    print(f"    Total time: {total_time:.1f}s")
    print(f"    Avg time:   {total_time/len(problems):.2f}s/problem")

    return {
        "mode": mode,
        "pass_at_1": pass_at_1,
        "total_time": total_time,
        "num_problems": len(problems),
        "sample_file": sample_file,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default=r"C:\Project\models\qwen3-0.6b\cpu_and_mobile\v4")
    parser.add_argument("-n", "--num_problems", type=int, default=None,
                        help="Number of problems to evaluate (default: all 164)")
    args = parser.parse_args()

    baseline = run_humaneval(args.model, speculative=False, max_problems=args.num_problems)
    speculative = run_humaneval(args.model, speculative=True, max_problems=args.num_problems)

    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Metric':<20} {'Baseline':>12} {'Speculative':>12} {'Delta':>12}")
    print(f"  {'-'*56}")
    print(f"  {'pass@1':<20} {baseline['pass_at_1']:>11.1%} {speculative['pass_at_1']:>11.1%} {speculative['pass_at_1']-baseline['pass_at_1']:>+11.1%}")
    print(f"  {'Total time':<20} {baseline['total_time']:>11.1f}s {speculative['total_time']:>11.1f}s {baseline['total_time']/speculative['total_time']:>11.2f}x")
    print(f"  {'Avg time/problem':<20} {baseline['total_time']/baseline['num_problems']:>11.2f}s {speculative['total_time']/speculative['num_problems']:>11.2f}s")

    if abs(speculative['pass_at_1'] - baseline['pass_at_1']) < 0.02:
        print(f"\n  ✓ No quality regression detected (delta < 2%)")
    else:
        print(f"\n  ⚠ Quality difference detected!")
