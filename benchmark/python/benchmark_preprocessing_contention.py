# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Measure CPU preprocessing alongside real CPU inference in separate processes."""

import argparse
import json
import multiprocessing
import os
from pathlib import Path
import statistics
import tempfile
import time

import numpy as np
import onnxruntime_genai as og


def cpu_model(model_path, threads, temporary):
    destination = Path(temporary)
    source = Path(model_path).resolve()
    config = json.loads((source / "genai_config.json").read_text())

    def configure_sessions(value):
        if isinstance(value, dict):
            if "filename" in value:
                options = value.setdefault("session_options", {})
                options.update(intra_op_num_threads=threads, inter_op_num_threads=1, provider_options=[])
                options["session.intra_op.allow_spinning"] = "0"
                options["session.inter_op.allow_spinning"] = "0"
            for child in value.values():
                configure_sessions(child)
        elif isinstance(value, list):
            for child in value:
                configure_sessions(child)

    configure_sessions(config["model"])
    (destination / "genai_config.json").write_text(json.dumps(config))
    for file in source.iterdir():
        if file.name != "genai_config.json":
            (destination / file.name).symlink_to(file)
    return og.Model(str(destination))


def summarize(samples, units, started, ended):
    elapsed = ended - started
    return {
        "operations": len(samples),
        "units": units,
        "elapsed_s": elapsed,
        "units_per_second": units / elapsed,
        "median_ms": statistics.median(samples),
        "p95_ms": float(np.percentile(samples, 95)),
    }


def inference_worker(connection, args, cpus):
    os.sched_setaffinity(0, cpus)
    with tempfile.TemporaryDirectory(prefix="genai-contention-") as temporary:
        model = cpu_model(args.inference_model, len(cpus), temporary)
        tokenizer = og.Tokenizer(model)
        input_ids = tokenizer.encode(args.inference_prompt)
        params = og.GeneratorParams(model)
        params.set_search_options(max_length=len(input_ids) + 32, do_sample=False)

        def generate(deadline):
            samples = []
            while time.perf_counter() < deadline:
                generator = og.Generator(model, params)
                generator.append_tokens(input_ids)
                while not generator.is_done() and time.perf_counter() < deadline:
                    start = time.perf_counter()
                    generator.generate_next_token()
                    samples.append((time.perf_counter() - start) * 1000)
                del generator
            return samples

        warmup = generate(time.perf_counter() + 2)
        if not warmup:
            raise RuntimeError("Real model warmup produced no tokens")
        connection.send({"ready": True, "inference_threads": len(cpus)})
        while (request := connection.recv()) is not None:
            start, deadline = request
            time.sleep(max(0, start - time.perf_counter()))
            started = time.perf_counter()
            samples = generate(deadline)
            connection.send(summarize(samples, len(samples), started, time.perf_counter()))


def preprocessing_worker(connection, args, cpus, workers):
    os.sched_setaffinity(0, cpus)
    os.environ["ORTGENAI_PREPROCESSING_THREADS"] = str(workers)
    temporary = tempfile.TemporaryDirectory(prefix="genai-preprocessing-contention-")
    model = cpu_model(args.preprocessing_model, 1, temporary.name)
    before = len(list(Path("/proc/self/task").iterdir()))
    processor = model.create_multimodal_processor()
    created = len(list(Path("/proc/self/task").iterdir())) - before
    if created != workers:
        raise RuntimeError(f"Expected {workers} preprocessing workers, observed {created}")
    images = og.Images.open(args.image)
    reference = processor(args.image_prompt, images=images)
    expected = {name: reference[name].as_numpy().copy() for name in reference.keys()}
    for _ in range(3):
        processor(args.image_prompt, images=images)
    connection.send({"ready": True, "preprocessing_workers": created})
    while (request := connection.recv()) is not None:
        start, deadline = request
        time.sleep(max(0, start - time.perf_counter()))
        started = time.perf_counter()
        samples = []
        output = None
        while time.perf_counter() < deadline:
            begin = time.perf_counter()
            output = processor(args.image_prompt, images=images)
            samples.append((time.perf_counter() - begin) * 1000)
        ended = time.perf_counter()
        if output is None or set(output.keys()) != set(expected):
            raise RuntimeError("Missing preprocessing outputs")
        for name, values in expected.items():
            actual = output[name].as_numpy()
            if actual.dtype != values.dtype or actual.shape != values.shape or actual.tobytes() != values.tobytes():
                raise RuntimeError(f"Preprocessing output changed under contention: {name}")
        connection.send(summarize(samples, len(samples), started, ended))
    del processor, model
    temporary.cleanup()


def receive(connection, process, timeout=300):
    if not connection.poll(timeout):
        raise RuntimeError(f"Timed out waiting for {process.name}; exit code: {process.exitcode}")
    return connection.recv()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-model", required=True)
    parser.add_argument("--preprocessing-model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--image-prompt", default="<image>Describe this image.")
    parser.add_argument("--inference-prompt", default="Explain why the sky looks blue during the day.")
    parser.add_argument("--cpus", required=True, help="Comma-separated logical CPUs shared by both processes")
    parser.add_argument("--cpu-counts", nargs="+", type=int, default=[4, 1])
    parser.add_argument("--workers", nargs="+", type=int, choices=range(4), default=[0, 1, 3])
    parser.add_argument("--seconds", type=float, default=6)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cpus = [int(cpu) for cpu in args.cpus.split(",")]
    if not set(cpus).issubset(os.sched_getaffinity(0)):
        parser.error("--cpus must be within the current CPU affinity")
    if any(count < 1 or count > len(cpus) for count in args.cpu_counts):
        parser.error("--cpu-counts must be positive and no larger than --cpus")
    if args.seconds <= 0 or args.trials < 1:
        parser.error("--seconds and --trials must be positive")
    context = multiprocessing.get_context("spawn")
    result = {
        "binding": og.__file__,
        "inference_model": str(Path(args.inference_model).resolve()),
        "preprocessing_model": str(Path(args.preprocessing_model).resolve()),
        "seconds_per_trial": args.seconds,
        "records": [],
    }
    for count in args.cpu_counts:
        selected = cpus[:count]
        # Rotate worker order across CPU allocations to reduce systematic order bias.
        worker_order = args.workers if count == args.cpu_counts[0] else list(reversed(args.workers))
        for workers in worker_order:
            processes = []
            connections = []
            try:
                for target, extra in ((inference_worker, ()), (preprocessing_worker, (workers,))):
                    parent, child = context.Pipe()
                    process = context.Process(
                        target=target, args=(child, args, selected, *extra), name=target.__name__
                    )
                    process.start()
                    child.close()
                    processes.append(process)
                    connections.append(parent)
                metadata = [receive(pipe, process) for pipe, process in zip(connections, processes)]
                for trial in range(args.trials):
                    modes = ["inference_only", "preprocessing_only", "concurrent"]
                    modes = modes[trial % 3:] + modes[:trial % 3]
                    for mode in modes:
                        active = [0, 1] if mode == "concurrent" else [0 if mode == "inference_only" else 1]
                        start = time.perf_counter() + 0.1
                        for index in active:
                            connections[index].send((start, start + args.seconds))
                        measurements = {
                            ("inference" if index == 0 else "preprocessing"):
                                receive(connections[index], processes[index])
                            for index in active
                        }
                        record = {
                            "cpus": selected, "workers": workers, "trial": trial + 1,
                            "mode": mode, "metadata": metadata, **measurements,
                        }
                        result["records"].append(record)
                        args.output.write_text(json.dumps(result, indent=2) + "\n")
                        print(json.dumps(record), flush=True)
            finally:
                for connection, process in zip(connections, processes):
                    if process.is_alive():
                        connection.send(None)
                    process.join(timeout=10)
                    if process.is_alive():
                        process.terminate()
                        process.join()
                    connection.close()
                failures = [process.exitcode for process in processes if process.exitcode != 0]
                if failures:
                    raise RuntimeError(f"Benchmark workers failed: {failures}")


if __name__ == "__main__":
    main()
