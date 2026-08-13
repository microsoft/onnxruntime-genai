# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Dataset loaders and quality evaluators shared by speculative benchmarks."""

from __future__ import annotations

import ast
import base64
import importlib.util
import io
import json
import multiprocessing
import os
import pickle
import re
import shutil
import string
import subprocess
import sys
import tempfile
import urllib.request
import zlib
import zipfile
from collections import Counter
from decimal import Decimal, InvalidOperation


GSM8K_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/master/"
    "grade_school_math/data/test.jsonl"
)
IFEVAL_REVISION = "966cd89545d6b6acfd7638bc708b98261ca58e84"
IFEVAL_URL = (
    "https://huggingface.co/datasets/google/IFEval/resolve/"
    f"{IFEVAL_REVISION}/ifeval_input_data.jsonl"
)
MATH500_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
MATH500_URL = (
    "https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/"
    f"{MATH500_REVISION}/test.jsonl"
)
LONGBENCH_REVISION = "5e628be450b7e67fb7ae6e201bd6d8f7056f7672"
LONGBENCH_URL = (
    "https://huggingface.co/datasets/zai-org/LongBench/resolve/"
    f"{LONGBENCH_REVISION}/data.zip"
)
BIGCODEBENCH_REVISION = "298d2cc7b96612e15e47313c3603ee124cee0c1f"
BIGCODEBENCH_VERSION = "v0.1.4"
LIVECODEBENCH_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
LIVECODEBENCH_BASE_URL = (
    "https://huggingface.co/datasets/livecodebench/code_generation_lite/"
    f"resolve/{LIVECODEBENCH_REVISION}"
)

SUPPORTED_SUITES = {
    "mtbench",
    "gsm8k",
    "ifeval",
    "math500",
    "longbench",
    "humaneval",
    "humanevalplus",
    "mbppplus",
    # "bigcodebench",  # Temporarily disabled pending a compatible evaluator environment.
    "livecodebench",
}
SUITE_ALIASES = {
    "math-500": "math500",
    "humaneval+": "humanevalplus",
    "human-eval-plus": "humanevalplus",
    "mbpp+": "mbppplus",
    "mbpp-plus": "mbppplus",
    # "bigcodebench-hard": "bigcodebench",
    "live-code-bench": "livecodebench",
}
CODE_EXECUTION_SUITES = {
    "humaneval",
    "humanevalplus",
    "mbppplus",
    # "bigcodebench",
    "livecodebench",
}
LONG_BENCH_TASKS = {
    "qasper": {
        "metric": "longbench_qa_f1",
        "max_output_tokens": 128,
        "prompt": (
            "You are given a scientific article and a question. Answer the question "
            "as concisely as you can, using a single phrase or sentence if possible. "
            'If it cannot be answered from the article, write "unanswerable". '
            'For yes/no questions, answer "yes", "no", or "unanswerable". Do not '
            "provide any explanation.\n\nArticle: {context}\n\nQuestion: {input}\n\nAnswer:"
        ),
    },
    "hotpotqa": {
        "metric": "longbench_qa_f1",
        "max_output_tokens": 32,
        "prompt": (
            "Answer the question based on the given passages. Only give the answer "
            "and do not output any other words.\n\nPassages:\n{context}\n\n"
            "Question: {input}\nAnswer:"
        ),
    },
    "gov_report": {
        "metric": "longbench_rouge_l",
        "max_output_tokens": 512,
        "prompt": (
            "You are given a report by a government agency. Write a one-page summary "
            "of the report.\n\nReport:\n{context}\n\nSummary:"
        ),
    },
    "passage_retrieval_en": {
        "metric": "longbench_retrieval",
        "max_output_tokens": 32,
        "prompt": (
            "Here are 30 paragraphs from Wikipedia, along with an abstract. Determine "
            "which paragraph the abstract is from.\n\n{context}\n\nAbstract:\n{input}\n\n"
            'Answer using only "Paragraph 1", "Paragraph 2", etc.\n\nThe answer is:'
        ),
    },
}
_IFEVAL_VENDOR_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "third_party", "ifeval"
)
_LIVECODEBENCH_IMPORTS = """from string import *
from re import *
from datetime import *
from collections import *
from heapq import *
from bisect import *
from copy import *
from math import *
from random import *
from statistics import *
from itertools import *
from functools import *
from operator import *
from io import *
from sys import *
from json import *
from builtins import *
from typing import *
import string
import re
import datetime
import collections
import heapq
import bisect
import copy
import math
import random
import statistics
import itertools
import functools
import operator
import io
import sys
import json
sys.setrecursionlimit(50000)
"""

GSM8K_FEW_SHOT_EXAMPLES = """Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
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

_EVALPLUS_EXPECTED = {}


def parse_suites(parser, value):
    suites = []
    for item in value.split(","):
        item = item.strip().lower()
        if item:
            suites.append(SUITE_ALIASES.get(item, item))
    if not suites:
        parser.error("--suites must contain at least one suite")
    unknown = sorted(set(suites) - SUPPORTED_SUITES)
    if unknown:
        parser.error(
            f"--suites contains unsupported values: {', '.join(unknown)}; "
            f"choose from {', '.join(sorted(SUPPORTED_SUITES))}"
        )
    return list(dict.fromkeys(suites))


def parse_longbench_tasks(parser, value):
    tasks = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not tasks:
        parser.error("--longbench-tasks must contain at least one task")
    unknown = sorted(set(tasks) - LONG_BENCH_TASKS.keys())
    if unknown:
        parser.error(
            f"--longbench-tasks contains unsupported values: {', '.join(unknown)}; "
            f"choose from {', '.join(LONG_BENCH_TASKS)}"
        )
    return list(dict.fromkeys(tasks))


def _download(url, path, name):
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    print(f"Downloading {name} to {path} ...")
    temporary_path = f"{path}.tmp.{os.getpid()}"
    try:
        urllib.request.urlretrieve(url, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def _require(module, purpose, installation):
    try:
        return __import__(module, fromlist=["*"])
    except ImportError as error:
        raise RuntimeError(
            f"{purpose} requires '{module}'. Install it with `{installation}`."
        ) from error


def empty_quality_result():
    return {
        "quality_metric": "",
        "quality_score_type": "",
        "quality_score": "",
        "quality_prediction": "",
        "quality_reference_answer": "",
        "quality_detail": "",
    }


def binary_quality(metric, passed, prediction="", reference="", detail=""):
    return {
        "quality_metric": metric,
        "quality_score_type": "binary",
        "quality_score": bool(passed),
        "quality_prediction": prediction,
        "quality_reference_answer": reference,
        "quality_detail": detail,
    }


def continuous_quality(metric, score, prediction="", reference="", detail=""):
    return {
        "quality_metric": metric,
        "quality_score_type": "continuous",
        "quality_score": float(score),
        "quality_prediction": prediction,
        "quality_reference_answer": reference,
        "quality_detail": detail,
    }


def extract_gsm8k_answer(text):
    match = re.search(r"####\s*([+-]?[\d,]+\.?\d*)", text)
    if match:
        return match.group(1).replace(",", "")
    match = re.search(
        r"(?:the answer is|answer:|= )\s*\$?([+-]?[\d,]+\.?\d*)",
        text,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).replace(",", "")
    numbers = re.findall(r"[+-]?\d[\d,]*\.?\d*", text)
    return numbers[-1].replace(",", "") if numbers else None


def trim_gsm8k_completion(completion):
    for stop in ("\nQuestion:", "\n\nQuestion", "\n\n\n"):
        index = completion.find(stop)
        if index != -1:
            completion = completion[:index]
    return completion


def gsm8k_answers_equal(prediction, reference):
    if prediction is None:
        return False
    try:
        return Decimal(prediction) == Decimal(reference)
    except InvalidOperation:
        return False


def load_gsm8k_prompts(dataset_path, max_problems):
    _download(GSM8K_URL, dataset_path, "GSM8K")
    examples = []
    with open(dataset_path, encoding="utf-8") as file:
        for index, line in enumerate(file):
            example = json.loads(line)
            ground_truth = extract_gsm8k_answer(example["answer"])
            if ground_truth is None:
                raise ValueError(f"GSM8K item {index} has no ground-truth answer")
            examples.append({
                "task": "gsm8k",
                "subcategory": "math_reasoning",
                "question_id": index,
                "text": (
                    f"{GSM8K_FEW_SHOT_EXAMPLES}Question: "
                    f"{example['question']}\nAnswer:"
                ),
                "raw_prompt": True,
                "quality_metric": "gsm8k_accuracy",
                "quality_score_type": "binary",
                "reference_answer": ground_truth,
                "max_output_tokens": 512,
            })
            if max_problems and len(examples) >= max_problems:
                break
    return examples


def load_humaneval_prompts(max_problems):
    try:
        from human_eval.data import read_problems
    except ImportError as error:
        raise RuntimeError(
            "HumanEval requires 'human-eval'. Install it with "
            "`pip install human-eval==1.0.3`."
        ) from error
    prompts = []
    for task_id, problem in read_problems().items():
        prompts.append({
            "task": "humaneval",
            "subcategory": "coding",
            "question_id": task_id,
            "text": problem["prompt"],
            "raw_prompt": True,
            "quality_metric": "humaneval_pass@1",
            "quality_score_type": "binary",
            "humaneval_problem": problem,
            "max_output_tokens": 512,
        })
        if max_problems and len(prompts) >= max_problems:
            break
    return prompts


def load_ifeval_prompts(dataset_path, max_problems):
    _download(IFEVAL_URL, dataset_path, "IFEval")
    prompts = []
    with open(dataset_path, encoding="utf-8") as file:
        for line in file:
            example = json.loads(line)
            prompts.append({
                "task": "ifeval",
                "subcategory": "instruction_following",
                "question_id": example["key"],
                "text": example["prompt"],
                "quality_metric": "ifeval_strict_prompt_accuracy",
                "quality_score_type": "binary",
                "ifeval_example": example,
                "max_output_tokens": 512,
            })
            if max_problems and len(prompts) >= max_problems:
                break
    return prompts


def load_math500_prompts(dataset_path, max_problems):
    _download(MATH500_URL, dataset_path, "MATH-500")
    prompts = []
    with open(dataset_path, encoding="utf-8") as file:
        for line in file:
            example = json.loads(line)
            prompts.append({
                "task": "math500",
                "subcategory": str(example["subject"]).lower().replace(" ", "_"),
                "question_id": example["unique_id"],
                "text": (
                    "Solve the following math problem efficiently and clearly. Think "
                    "step by step. The last line must be: Therefore, the final answer "
                    "is: $\\boxed{ANSWER}$.\n\n" + example["problem"]
                ),
                "quality_metric": "math500_accuracy",
                "quality_score_type": "binary",
                "reference_answer": example["answer"],
                "max_output_tokens": 512,
            })
            if max_problems and len(prompts) >= max_problems:
                break
    return prompts


def _load_dataset(*args, **kwargs):
    datasets = _require(
        "datasets",
        "Hugging Face benchmark datasets",
        "pip install datasets==2.21.0",
    )
    return datasets.load_dataset(*args, **kwargs)


def load_longbench_prompts(tasks, max_problems_per_task, max_input_tokens):
    archive_path = os.path.join(
        os.path.expanduser("~"),
        ".cache",
        "onnxruntime-genai",
        "longbench",
        LONGBENCH_REVISION,
        "data.zip",
    )
    _download(LONGBENCH_URL, archive_path, "LongBench")
    prompts = []
    with zipfile.ZipFile(archive_path) as archive:
        for task in tasks:
            config = LONG_BENCH_TASKS[task]
            with archive.open(f"data/{task}.jsonl") as raw_file:
                file = io.TextIOWrapper(raw_file, encoding="utf-8")
                for index, line in enumerate(file):
                    example = json.loads(line)
                    prompts.append({
                        "task": f"longbench/{task}",
                        "subcategory": "long_context",
                        "question_id": index,
                        "text": config["prompt"].format(
                            context=example["context"],
                            input=example["input"],
                        ),
                        "quality_metric": config["metric"],
                        "quality_score_type": "continuous",
                        "reference_answers": list(example["answers"]),
                        "all_classes": list(example.get("all_classes") or []),
                        "max_input_tokens": max_input_tokens,
                        "max_output_tokens": config["max_output_tokens"],
                        "longbench_task": task,
                    })
                    if (
                        max_problems_per_task
                        and index + 1 >= max_problems_per_task
                    ):
                        break
    return prompts


def _load_evalplus(dataset_name, max_problems):
    try:
        if dataset_name == "humaneval":
            from evalplus.data import get_human_eval_plus as loader
        else:
            from evalplus.data import get_mbpp_plus as loader
    except ImportError as error:
        raise RuntimeError(
            f"{dataset_name}+ requires EvalPlus. Install it with "
            "`pip install evalplus==0.3.1`."
        ) from error
    task_name = f"{dataset_name}plus"
    prompts = []
    for task_id, problem in loader().items():
        prompts.append({
            "task": task_name,
            "subcategory": "coding",
            "question_id": task_id,
            "text": problem["prompt"],
            "raw_prompt": True,
            "quality_metric": f"{task_name}_pass@1",
            "quality_score_type": "binary",
            "evalplus_dataset": dataset_name,
            "evalplus_problem": problem,
            "max_output_tokens": 512,
        })
        if max_problems and len(prompts) >= max_problems:
            break
    return prompts


def load_humanevalplus_prompts(max_problems):
    return _load_evalplus("humaneval", max_problems)


def load_mbppplus_prompts(max_problems):
    return _load_evalplus("mbpp", max_problems)


def _parse_library_list(value):
    if not value:
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else [str(parsed)]
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple)):
                return [str(item) for item in parsed]
        except (SyntaxError, ValueError):
            pass
        return [item.strip() for item in value.split(",") if item.strip()]


def _missing_modules(module_names):
    return sorted({
        name
        for name in module_names
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", name)
        and importlib.util.find_spec(name.split(".")[0]) is None
    })


def load_bigcodebench_prompts(max_problems, subset):
    repository = (
        "bigcode/bigcodebench-hard" if subset == "hard"
        else "bigcode/bigcodebench"
    )
    dataset = _load_dataset(
        repository,
        split=BIGCODEBENCH_VERSION,
        revision=BIGCODEBENCH_REVISION if subset == "hard" else None,
    )
    prompts = []
    for example in dataset:
        prompts.append({
            "task": f"bigcodebench/{subset}",
            "subcategory": "coding",
            "question_id": example["task_id"],
            "text": example["instruct_prompt"],
            "quality_metric": "bigcodebench_pass@1",
            "quality_score_type": "binary",
            "bigcodebench_problem": dict(example),
            "required_libraries": _parse_library_list(example.get("libs")),
            "max_output_tokens": 1024,
        })
        if max_problems and len(prompts) >= max_problems:
            break
    missing = _missing_modules(
        library
        for prompt in prompts
        for library in prompt["required_libraries"]
    )
    if missing:
        raise RuntimeError(
            "BigCodeBench task dependencies are missing: "
            f"{', '.join(missing)}. Run this suite in the official "
            "BigCodeBench evaluation image or provision its pinned "
            "Requirements/requirements-eval.txt on an isolated agent."
        )
    return prompts


def _decode_livecodebench_private_tests(value):
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return json.loads(
            pickle.loads(zlib.decompress(base64.b64decode(value.encode("utf-8"))))
        )


def _livecodebench_file_numbers(release):
    if release == "release_latest":
        return list(range(1, 7))
    match = re.fullmatch(r"release_v([1-6])", release)
    if match:
        return list(range(1, int(match.group(1)) + 1))
    match = re.fullmatch(r"v([1-6])", release)
    if match:
        return [int(match.group(1))]
    match = re.fullmatch(r"v([1-6])_v([1-6])", release)
    if match and int(match.group(1)) <= int(match.group(2)):
        return list(range(int(match.group(1)), int(match.group(2)) + 1))
    raise ValueError(
        "LiveCodeBench release must be release_v1..release_v6, "
        "release_latest, v1..v6, or a range such as v5_v6"
    )


def load_livecodebench_prompts(max_problems, release):
    cache_directory = os.path.join(
        os.path.expanduser("~"),
        ".cache",
        "onnxruntime-genai",
        "livecodebench",
        LIVECODEBENCH_REVISION,
    )
    examples = []
    file_numbers = _livecodebench_file_numbers(release)
    for number in reversed(file_numbers):
        filename = "test.jsonl" if number == 1 else f"test{number}.jsonl"
        path = os.path.join(cache_directory, filename)
        _download(
            f"{LIVECODEBENCH_BASE_URL}/{filename}",
            path,
            f"LiveCodeBench {filename}",
        )
        with open(path, encoding="utf-8") as file:
            examples.extend(json.loads(line) for line in file if line.strip())
        if max_problems and len(examples) >= max_problems:
            break
    if max_problems:
        examples = sorted(
            examples,
            key=lambda example: (
                example["contest_date"],
                example["question_id"],
            ),
            reverse=True,
        )[:max_problems]
    prompts = []
    for example in examples:
        public_tests = json.loads(example["public_test_cases"])
        private_tests = _decode_livecodebench_private_tests(
            example["private_test_cases"]
        )
        metadata = json.loads(example["metadata"])
        tests = public_tests + private_tests
        evaluation_sample = {
            "input_output": json.dumps({
                "inputs": [test["input"] for test in tests],
                "outputs": [test["output"] for test in tests],
                "fn_name": metadata.get("func_name"),
            })
        }
        if example["starter_code"]:
            format_instruction = (
                "Use the following starter code and enclose the complete solution "
                f"in a Python code block:\n```python\n{example['starter_code']}\n```"
            )
        else:
            format_instruction = (
                "Read from stdin and write to stdout. Enclose the complete solution "
                "in a Python code block."
            )
        prompts.append({
            "task": "livecodebench",
            "subcategory": str(example["difficulty"]),
            "question_id": example["question_id"],
            "text": (
                "You are an expert Python programmer. Generate a correct program "
                "that matches the specification and passes all tests.\n\n"
                f"### Question:\n{example['question_content']}\n\n"
                f"### Format:\n{format_instruction}\n\n### Answer:"
            ),
            "quality_metric": "livecodebench_pass@1",
            "quality_score_type": "binary",
            "livecodebench_sample": evaluation_sample,
            "livecodebench_release": release,
            "max_output_tokens": 1024,
        })
    return prompts


def load_additional_suite_prompts(args, suites):
    prompts = []
    if "gsm8k" in suites:
        prompts.extend(load_gsm8k_prompts(args.gsm8k_path, args.gsm8k_problems))
    if "ifeval" in suites:
        prompts.extend(load_ifeval_prompts(args.ifeval_path, args.ifeval_problems))
    if "math500" in suites:
        prompts.extend(load_math500_prompts(args.math500_path, args.math500_problems))
    if "longbench" in suites:
        prompts.extend(load_longbench_prompts(
            args.longbench_tasks,
            args.longbench_problems_per_task,
            args.longbench_max_input_tokens,
        ))
    if "humaneval" in suites:
        prompts.extend(load_humaneval_prompts(args.humaneval_problems))
    if "humanevalplus" in suites:
        prompts.extend(load_humanevalplus_prompts(args.humanevalplus_problems))
    if "mbppplus" in suites:
        prompts.extend(load_mbppplus_prompts(args.mbppplus_problems))
    # BigCodeBench is temporarily disabled; keep its loader dormant for later use.
    # if "bigcodebench" in suites:
    #     prompts.extend(load_bigcodebench_prompts(
    #         args.bigcodebench_problems, args.bigcodebench_subset
    #     ))
    if "livecodebench" in suites:
        prompts.extend(load_livecodebench_prompts(
            args.livecodebench_problems, args.livecodebench_release
        ))
    return prompts


def extract_code_block(completion):
    matches = re.findall(
        r"```(?:python|py)?\s*\n(.*?)```",
        completion,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if matches:
        return matches[-1].strip()
    return completion.strip()


def _longest_valid_python_prefix(source):
    lines = source.splitlines()
    for end in range(len(lines), 0, -1):
        candidate = "\n".join(lines[:end])
        try:
            ast.parse(candidate)
            return candidate
        except SyntaxError:
            continue
    return source


def _compose_function_solution(problem, completion):
    matches = re.findall(
        r"```(?:python|py)?\s*\n(.*?)```",
        completion,
        flags=re.IGNORECASE | re.DOTALL,
    )
    code = matches[-1].strip() if matches else completion.rstrip()
    entry_point = re.escape(problem["entry_point"])
    if re.search(rf"(?m)^def\s+{entry_point}\s*\(", code):
        solution = code
    else:
        solution = problem["prompt"] + code
    return _longest_valid_python_prefix(solution)


def _execute_humaneval(problem, completion, connection):
    from human_eval.execution import create_tempdir, reliability_guard, swallow_io

    try:
        with create_tempdir():
            rmtree = shutil.rmtree
            rmdir = os.rmdir
            chdir = os.chdir
            reliability_guard()
            check_program = (
                _compose_function_solution(problem, completion)
                + "\n"
                + problem["test"]
                + "\n"
                + f"check({problem['entry_point']})"
            )
            try:
                with swallow_io():
                    exec(check_program, {})
                result = "passed"
            except BaseException as error:
                result = f"failed: {error}"
            finally:
                shutil.rmtree = rmtree
                os.rmdir = rmdir
                os.chdir = chdir
        connection.send(result)
    except BaseException as error:
        connection.send(f"failed: {error}")
    finally:
        connection.close()


def check_humaneval_correctness(problem, completion, timeout):
    receive, send = multiprocessing.Pipe(duplex=False)
    process = multiprocessing.Process(
        target=_execute_humaneval,
        args=(problem, completion, send),
    )
    process.start()
    send.close()
    process.join(timeout)
    if process.is_alive():
        process.kill()
        process.join()
        result = "timed out"
    elif receive.poll():
        result = receive.recv()
    else:
        result = f"failed: evaluator process exited with code {process.exitcode}"
    receive.close()
    return result == "passed", result


def _score_ifeval(item, completion):
    if _IFEVAL_VENDOR_ROOT not in sys.path:
        sys.path.insert(0, _IFEVAL_VENDOR_ROOT)
    try:
        from instruction_following_eval import evaluation_lib
    except ImportError as error:
        raise RuntimeError(
            "IFEval requires the pinned official instruction_following_eval sources "
            "vendored beside this script."
        ) from error
    example = item["ifeval_example"]
    inp = evaluation_lib.InputExample(
        key=example["key"],
        instruction_id_list=example["instruction_id_list"],
        prompt=example["prompt"],
        kwargs=example["kwargs"],
    )
    responses = {example["prompt"]: completion}
    strict = evaluation_lib.test_instruction_following_strict(inp, responses)
    loose = evaluation_lib.test_instruction_following_loose(inp, responses)
    detail = json.dumps({
        "strict_instruction_accuracy": (
            sum(strict.follow_instruction_list) / len(strict.follow_instruction_list)
        ),
        "strict_instructions": strict.follow_instruction_list,
        "loose_prompt_accuracy": loose.follow_all_instructions,
        "loose_instruction_accuracy": (
            sum(loose.follow_instruction_list) / len(loose.follow_instruction_list)
        ),
        "loose_instructions": loose.follow_instruction_list,
    })
    return binary_quality(
        item["quality_metric"],
        strict.follow_all_instructions,
        detail=detail,
    )


def _math500_worker(reference, completion, connection):
    try:
        import logging

        logging.getLogger("math_verify").setLevel(logging.ERROR)
        from math_verify import LatexExtractionConfig, parse, verify
        config = [LatexExtractionConfig()]
        gold = parse(
            f"${reference}$", extraction_config=config, parsing_timeout=0
        )
        prediction = parse(
            completion, extraction_config=config, parsing_timeout=0
        )
        passed = bool(verify(gold, prediction, timeout_seconds=0))
        connection.send((
            passed,
            " | ".join(str(value) for value in prediction),
            "",
        ))
    except BaseException as error:
        connection.send((False, "", f"failed: {error}"))
    finally:
        connection.close()


def _score_math500(item, completion, timeout):
    try:
        __import__("math_verify")
    except ImportError as error:
        raise RuntimeError(
            "MATH-500 requires math-verify. Install it with "
            "`pip install 'math-verify[antlr4_13_2]==0.9.0'`."
        ) from error
    reference = item["reference_answer"]
    receive, send = multiprocessing.Pipe(duplex=False)
    process = multiprocessing.Process(
        target=_math500_worker,
        args=(reference, completion, send),
    )
    process.start()
    send.close()
    process.join(max(10.0, timeout))
    if process.is_alive():
        process.kill()
        process.join()
        passed, prediction, detail = False, "", "timed out"
    elif receive.poll():
        passed, prediction, detail = receive.recv()
    else:
        passed, prediction, detail = (
            False,
            "",
            f"evaluator process exited with code {process.exitcode}",
        )
    receive.close()
    return binary_quality(
        item["quality_metric"],
        passed,
        prediction=prediction,
        reference=reference,
        detail=detail,
    )


def _normalize_answer(text):
    def remove_articles(value):
        return re.sub(r"\b(a|an|the)\b", " ", value)

    exclude = set(string.punctuation)
    text = "".join(character for character in text.lower() if character not in exclude)
    return " ".join(remove_articles(text).split())


def _qa_f1(prediction, reference):
    prediction_tokens = _normalize_answer(prediction).split()
    reference_tokens = _normalize_answer(reference).split()
    common = Counter(prediction_tokens) & Counter(reference_tokens)
    same = sum(common.values())
    if not same:
        return 0.0
    precision = same / len(prediction_tokens)
    recall = same / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def _longbench_score(task, prediction, reference):
    if task in {"qasper", "hotpotqa"}:
        return _qa_f1(prediction, reference)
    if task == "gov_report":
        rouge = _require(
            "rouge",
            "LongBench summarization scoring",
            "pip install rouge==1.0.1",
        )
        try:
            return float(
                rouge.Rouge().get_scores([prediction], [reference], avg=True)
                ["rouge-l"]["f"]
            )
        except ValueError:
            return 0.0
    if task == "passage_retrieval_en":
        match = re.search(r"Paragraph (\d+)", reference)
        if not match:
            raise ValueError(f"LongBench retrieval reference is malformed: {reference!r}")
        numbers = re.findall(r"Paragraph (\d+)", prediction)
        return (
            sum(number == match.group(1) for number in numbers) / len(numbers)
            if numbers else 0.0
        )
    raise ValueError(f"Unsupported LongBench task: {task}")


def _score_longbench(item, completion):
    answers = item["reference_answers"]
    score = max(
        _longbench_score(item["longbench_task"], completion, answer)
        for answer in answers
    )
    return continuous_quality(
        item["quality_metric"],
        score,
        prediction=completion[:500],
        reference=json.dumps(answers, ensure_ascii=False),
    )


def _evalplus_expected(dataset, problem):
    key = (dataset, problem["task_id"])
    if key in _EVALPLUS_EXPECTED:
        return _EVALPLUS_EXPECTED[key]
    from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
    from evalplus.gen.util import trusted_exec

    output_not_none = (
        dataset == "mbpp"
        and problem["entry_point"] in MBPP_OUTPUT_NOT_NONE_TASKS
    )
    result = {}
    for test_set in ("base", "plus"):
        result[test_set], result[f"{test_set}_time"] = trusted_exec(
            problem["prompt"] + problem["canonical_solution"],
            problem[f"{test_set}_input"],
            problem["entry_point"],
            record_time=True,
            output_not_none=output_not_none,
        )
    _EVALPLUS_EXPECTED[key] = result
    return result


def _evalplus_worker(
    dataset,
    solution,
    inputs,
    entry_point,
    expected,
    atol,
    connection,
):
    try:
        import numpy
        from evalplus.eval._special_oracle import (
            MBPP_OUTPUT_NOT_NONE_TASKS,
            MBPP_OUTPUT_SET_EQ_TASKS,
            _digit_distance_nums,
            _poly,
            _surface_Area,
        )

        namespace = {}
        exec(solution, namespace)
        function = namespace[entry_point]
        for index, (arguments, reference) in enumerate(zip(inputs, expected)):
            actual = function(*arguments)
            if dataset == "mbpp" and entry_point == "are_equivalent":
                passed = True
            elif dataset == "mbpp" and entry_point == "sum_div":
                passed = actual == reference or actual == 0
            elif dataset == "mbpp" and entry_point == "surface_Area":
                passed = (
                    actual == reference
                    or abs(actual - _surface_Area(*arguments)) <= atol
                )
            elif dataset == "mbpp" and entry_point == "digit_distance_nums":
                passed = (
                    actual == reference
                    or actual == _digit_distance_nums(*arguments)
                )
            elif dataset == "mbpp" and entry_point in MBPP_OUTPUT_SET_EQ_TASKS:
                passed = set(actual) == set(reference)
            elif dataset == "mbpp" and entry_point in MBPP_OUTPUT_NOT_NONE_TASKS:
                passed = (
                    actual == reference
                    if isinstance(actual, bool)
                    else reference == (actual is not None)
                )
            elif dataset == "humaneval" and entry_point == "find_zero":
                passed = abs(_poly(*arguments, actual)) <= atol
            else:
                exact = (
                    numpy.array_equal(actual, reference)
                    if isinstance(actual, numpy.ndarray)
                    or isinstance(reference, numpy.ndarray)
                    else actual == reference
                )
                passed = bool(exact)
                if not passed and atol:
                    if type(actual) is not type(reference):
                        passed = False
                    elif isinstance(reference, (list, tuple)) and (
                        len(actual) != len(reference)
                    ):
                        passed = False
                    else:
                        passed = bool(
                            numpy.allclose(actual, reference, rtol=1e-7, atol=atol)
                        )
                elif not passed and (
                    isinstance(reference, float)
                    or (
                        isinstance(reference, (list, tuple))
                        and reference
                        and all(isinstance(value, float) for value in reference)
                    )
                ):
                    passed = bool(
                        numpy.allclose(actual, reference, rtol=1e-7, atol=1e-6)
                    )
            if not passed:
                connection.send((
                    False,
                    f"wrong answer on test {index}: {actual!r} != {reference!r}",
                ))
                return
        connection.send((True, "passed"))
    except BaseException as error:
        connection.send((False, f"failed: {error}"))
    finally:
        connection.close()


def _run_evalplus_test_set(
    dataset,
    solution,
    inputs,
    entry_point,
    expected,
    atol,
    timeout,
):
    receive, send = multiprocessing.Pipe(duplex=False)
    process = multiprocessing.Process(
        target=_evalplus_worker,
        args=(
            dataset,
            solution,
            inputs,
            entry_point,
            expected,
            atol,
            send,
        ),
    )
    process.start()
    send.close()
    process.join(timeout)
    if process.is_alive():
        process.kill()
        process.join()
        result = (False, "timed out")
    elif receive.poll():
        result = receive.recv()
    else:
        result = (
            False,
            f"evaluator process exited with code {process.exitcode}",
        )
    receive.close()
    return result


def _score_evalplus(item, completion, timeout):
    try:
        __import__("evalplus")
    except ImportError as error:
        raise RuntimeError(
            "HumanEval+ and MBPP+ require EvalPlus. Install it with "
            "`pip install evalplus==0.3.1`."
        ) from error

    dataset = item["evalplus_dataset"]
    problem = item["evalplus_problem"]
    expected = _evalplus_expected(dataset, problem)
    solution = _compose_function_solution(problem, completion)
    statuses = {}
    for test_set in ("base", "plus"):
        passed, detail = _run_evalplus_test_set(
            dataset,
            solution,
            problem[f"{test_set}_input"],
            problem["entry_point"],
            expected[test_set],
            problem["atol"],
            timeout,
        )
        statuses[test_set] = detail
        if not passed:
            break
    return binary_quality(
        item["quality_metric"],
        all(status == "passed" for status in statuses.values())
        and len(statuses) == 2,
        detail=json.dumps(statuses),
    )


def _bigcodebench_worker(problem, solution, connection):
    try:
        import types
        import unittest

        module = types.ModuleType("__test__")
        exec(
            compile(solution + "\n" + problem["test"], "__test__.py", "exec"),
            module.__dict__,
        )
        test_cases = getattr(module, "TestCases")
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(test_cases)
        result = unittest.TestResult()
        suite.run(result)
        issues = result.failures + result.errors
        detail = "\n".join(
            f"{test.id()}: {trace}" for test, trace in issues
        )[-4000:]
        connection.send((not issues, detail))
    except BaseException as error:
        connection.send((False, f"failed: {error}"))
    finally:
        connection.close()


def _score_bigcodebench(item, completion, timeout):
    try:
        from bigcodebench.sanitize import sanitize
    except ImportError as error:
        raise RuntimeError(
            "BigCodeBench requires the official package. Install it with "
            "`pip install --no-deps bigcodebench==0.2.5`."
        ) from error
    missing = _missing_modules(item["required_libraries"])
    if missing:
        raise RuntimeError(
            "BigCodeBench evaluation dependencies are missing: "
            f"{', '.join(sorted(missing))}. Run in the official BigCodeBench "
            "evaluation image or provision its Requirements/requirements-eval.txt."
        )
    problem = item["bigcodebench_problem"]
    code = extract_code_block(completion)
    entry_point = re.escape(problem["entry_point"])
    if re.search(rf"(?m)^def\s+{entry_point}\s*\(", code):
        candidate = code
    else:
        candidate = problem["code_prompt"] + "\n    pass\n" + code
    solution = sanitize(candidate, problem["entry_point"])
    receive, send = multiprocessing.Pipe(duplex=False)
    process = multiprocessing.Process(
        target=_bigcodebench_worker,
        args=(problem, solution, send),
    )
    process.start()
    send.close()
    process.join(timeout)
    if process.is_alive():
        process.kill()
        process.join()
        passed, detail = False, "timed out"
    elif receive.poll():
        passed, detail = receive.recv()
    else:
        passed = False
        detail = f"evaluator process exited with code {process.exitcode}"
    receive.close()
    return binary_quality(
        item["quality_metric"],
        passed,
        detail=detail,
    )


def _decimal_lines_equal(actual, expected):
    actual_lines = [line.strip() for line in actual.strip().splitlines()]
    expected_lines = [line.strip() for line in expected.strip().splitlines()]
    if actual_lines == expected_lines:
        return True
    if len(actual_lines) != len(expected_lines):
        return False
    try:
        return all(
            [Decimal(value) for value in actual_line.split()]
            == [Decimal(value) for value in expected_line.split()]
            for actual_line, expected_line in zip(actual_lines, expected_lines)
        )
    except InvalidOperation:
        return False


def _livecodebench_worker(sample, code, timeout, connection):
    try:
        io_data = json.loads(sample["input_output"])
        inputs = io_data["inputs"]
        outputs = io_data["outputs"]
        function_name = io_data.get("fn_name")
        if function_name:
            namespace = {}
            exec(_LIVECODEBENCH_IMPORTS + "\n" + code, namespace)
            owner = namespace.get("Solution")
            callable_owner = owner() if owner is not None else namespace
            function = (
                getattr(callable_owner, function_name)
                if owner is not None else callable_owner[function_name]
            )
            for raw_input, raw_output in zip(inputs, outputs):
                if isinstance(raw_input, list):
                    raw_input = "\n".join(raw_input)
                arguments = [
                    json.loads(line) for line in raw_input.splitlines()
                ]
                actual = function(*arguments)
                expected = json.loads(raw_output)
                if isinstance(actual, tuple):
                    actual = list(actual)
                if actual != expected:
                    connection.send((False, "wrong answer"))
                    return
        else:
            for raw_input, raw_output in zip(inputs, outputs):
                if isinstance(raw_input, list):
                    raw_input = "\n".join(raw_input)
                with tempfile.TemporaryDirectory() as directory:
                    script = os.path.join(directory, "solution.py")
                    with open(script, "w", encoding="utf-8") as file:
                        file.write(_LIVECODEBENCH_IMPORTS + "\n" + code)
                    result = subprocess.run(
                        [sys.executable, "-I", script],
                        input=raw_input,
                        text=True,
                        capture_output=True,
                        cwd=directory,
                        timeout=timeout,
                        check=False,
                    )
                if result.returncode or not _decimal_lines_equal(
                    result.stdout, raw_output
                ):
                    detail = (
                        f"exit={result.returncode}; stderr={result.stderr[-500:]}"
                        if result.returncode else "wrong answer"
                    )
                    connection.send((False, detail))
                    return
        connection.send((True, "passed"))
    except subprocess.TimeoutExpired:
        connection.send((False, "timed out"))
    except BaseException as error:
        connection.send((False, f"failed: {error}"))
    finally:
        connection.close()


def _score_livecodebench(item, completion, timeout):
    code = extract_code_block(completion)
    receive, send = multiprocessing.Pipe(duplex=False)
    process = multiprocessing.Process(
        target=_livecodebench_worker,
        args=(item["livecodebench_sample"], code, timeout, send),
    )
    process.start()
    send.close()
    input_count = len(
        json.loads(item["livecodebench_sample"]["input_output"])["inputs"]
    )
    process.join((timeout + 1) * input_count + 5)
    if process.is_alive():
        process.kill()
        process.join()
        passed, detail = False, "global timeout"
    elif receive.poll():
        passed, detail = receive.recv()
    else:
        passed = False
        detail = f"evaluator process exited with code {process.exitcode}"
    receive.close()
    return binary_quality(item["quality_metric"], passed, detail=detail)


def score_completion(item, token_ids, timeout, cache, decode):
    metric = item.get("quality_metric")
    if not metric:
        return empty_quality_result()
    cache_key = (item["task"], str(item["question_id"]), tuple(token_ids))
    if cache_key in cache:
        return cache[cache_key]
    completion = decode(token_ids)
    if item["task"] == "gsm8k":
        prediction = extract_gsm8k_answer(trim_gsm8k_completion(completion))
        result = binary_quality(
            metric,
            gsm8k_answers_equal(prediction, item["reference_answer"]),
            prediction=prediction or "",
            reference=item["reference_answer"],
        )
    elif item["task"] == "humaneval":
        passed, detail = check_humaneval_correctness(
            item["humaneval_problem"],
            completion,
            timeout,
        )
        result = binary_quality(metric, passed, detail=detail)
    elif item["task"] == "ifeval":
        result = _score_ifeval(item, completion)
    elif item["task"] == "math500":
        result = _score_math500(item, completion, timeout)
    elif item["task"].startswith("longbench/"):
        result = _score_longbench(item, completion)
    elif item["task"] in {"humanevalplus", "mbppplus"}:
        result = _score_evalplus(item, completion, timeout)
    # elif item["task"].startswith("bigcodebench/"):
    #     result = _score_bigcodebench(item, completion, timeout)
    elif item["task"] == "livecodebench":
        result = _score_livecodebench(item, completion, timeout)
    else:
        raise ValueError(f"Unsupported quality task: {item['task']}")
    cache[cache_key] = result
    return result


def classify_quality_transition(baseline_quality, candidate_quality, candidate_name):
    if not baseline_quality["quality_metric"]:
        return ""
    score_type = baseline_quality["quality_score_type"]
    baseline_score = baseline_quality["quality_score"]
    candidate_score = candidate_quality["quality_score"]
    if score_type == "continuous":
        difference = float(candidate_score) - float(baseline_score)
        if abs(difference) <= 1e-12:
            return "equal_score"
        return (
            f"{candidate_name}_higher_score"
            if difference > 0 else "baseline_higher_score"
        )
    baseline_correct = bool(baseline_score)
    candidate_correct = bool(candidate_score)
    if baseline_correct and candidate_correct:
        return "both_correct"
    if baseline_correct:
        return "baseline_only"
    if candidate_correct:
        return f"{candidate_name}_only"
    baseline_prediction = baseline_quality["quality_prediction"]
    candidate_prediction = candidate_quality["quality_prediction"]
    if baseline_prediction and baseline_prediction == candidate_prediction:
        return "both_wrong_same_prediction"
    if baseline_prediction and candidate_prediction:
        return "both_wrong_different_prediction"
    return "both_wrong"


def format_quality(quality):
    if not quality["quality_metric"]:
        return "not_scored"
    prediction = quality["quality_prediction"]
    if quality["quality_score_type"] == "continuous":
        result = f"score={float(quality['quality_score']):.1%}"
    else:
        result = "correct" if quality["quality_score"] else "wrong"
    return f"{result}({prediction})" if prediction else result


def format_quality_with_reference(quality):
    result = format_quality(quality)
    reference = quality["quality_reference_answer"]
    return f"{result} expected={reference}" if reference else result


def format_quality_summary_lines(rows, baseline_label, candidate_label):
    groups = {}
    for row in rows:
        if row["quality_metric"] and row["rep"] == 0:
            key = (row["task"], row["quality_metric"])
            groups.setdefault(key, []).append(row)
    lines = []
    for (task, metric), task_rows in groups.items():
        transitions = Counter(
            row["quality_transition"] for row in task_rows
        )
        baseline_score = sum(
            float(row["baseline_quality_score"]) for row in task_rows
        ) / len(task_rows)
        candidate_score = sum(
            float(row["quality_score"]) for row in task_rows
        ) / len(task_rows)
        transition_text = ", ".join(
            f"{name}={value}" for name, value in sorted(transitions.items())
        )
        lines.append(
            f"quality {task} ({metric}): "
            f"{baseline_label}={baseline_score:.1%} "
            f"{candidate_label}={candidate_score:.1%} "
            f"delta={candidate_score - baseline_score:+.1%}; "
            f"{transition_text}"
        )
    return lines


def truncate_prompt_tokens(token_ids, item):
    limit = int(item.get("max_input_tokens") or 0)
    if not limit or len(token_ids) <= limit:
        return token_ids
    first = limit // 2
    return token_ids[:first] + token_ids[-(limit - first):]


def generation_limit(item, configured_limit):
    return int(item.get("max_output_tokens") or configured_limit)
