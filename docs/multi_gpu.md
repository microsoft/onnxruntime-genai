# Multi-GPU (tensor / expert parallel) models

A tensor-parallel export is one ONNX graph per GPU. Each graph holds a slice of every weight and
the ranks meet in the collective ops (`com.microsoft.AllReduce`) of every layer, so no rank can
make progress alone: they all have to be inside the same forward pass at the same time.

Only the `Generator` (static batch) path is supported today. `Engine` / paged attention is not.
Linux only.

## Why there is a worker process per rank

ONNX Runtime's NCCL context is a process-wide singleton, built from the `LOCAL_RANK` and
`LOCAL_WORLD_SIZE` environment variables the first time a collective kernel is created. Two ranks
cannot share a process, so onnxruntime-genai runs:

* **rank 0** in your process. It owns the tokenizer, the search, the sampling, the KV cache and
  the logits - everything a single-GPU model owns.
* **ranks 1..N-1** as `onnxruntime-genai-tp-worker` child processes. Each one runs an ordinary
  `Generator` over its own rank's graph and replays rank 0's forward passes on it. Because only
  rank 0 samples, the ranks cannot disagree about what was generated.

## Layout

```
model/
  genai_config.json
  tokenizer.json
  rank_0/model.onnx      rank_0/model.onnx.data
  rank_1/model.onnx      rank_1/model.onnx.data
  ...
```

## Configuration

```json
{
  "model": {
    "multi_gpu": {
      "world_size": 8
    }
  }
}
```

| Key | Default | Meaning |
| --- | --- | --- |
| `world_size` | `1` | Number of GPUs. `1` disables multi-GPU. |
| `rank_dir` | `"rank_%d"` | Subdirectory holding each rank's graph; `%d` is the rank. |
| `master_ip` | `"127.0.0.1"` | NCCL bootstrap address. Must be a literal address - ORT resolves it with `AF_UNSPEC`, so `localhost` can pick IPv6 on one side and IPv4 on the other. |
| `master_port` | `19555` | TCP port rank 0 uses to hand out the NCCL unique id. |
| `worker_executable` | next to `libonnxruntime-genai.so` | Path to `onnxruntime-genai-tp-worker`. Also settable with `ORTGENAI_TP_WORKER`. |
| `log_dir` | empty | If set, each worker's stdout/stderr goes to `<log_dir>/rank<N>.log`. Worth setting: it is the only place a worker's errors appear once it is running. |
| `startup_timeout_s` | `1800` | How long a worker waits for rank 0 to start listening. |

Rank *n* is pinned to GPU *n* through `CUDA_VISIBLE_DEVICES`. Rank 0 keeps whatever device the
host process was already using.

## Using it

Nothing in the API changes:

```python
model = og.Model("model")
generator = og.Generator(model, params)
```

Constraints: `batch_size` and `num_beams` must both be 1.

## Startup

Model loading is serialized, so expect roughly twice a single rank's load time:

1. Rank 0 spawns the workers. Each greets rank 0 and then waits for rank 0's bootstrap socket to
   appear, without loading anything.
2. Rank 0 loads its weights and, from inside session creation, binds the bootstrap port and blocks
   until every worker has connected.
3. The workers see the listening socket, load their own weights in parallel, and connect.

The wait in step 1 is deliberate. ORT's connect retry window is 40 seconds and is not
configurable, so a worker that starts connecting before rank 0 is listening would give up long
before rank 0 finished loading.

## When something goes wrong

Anything a worker can detect before it starts loading - a missing executable, an unreadable
config - is reported as an exception from `Model` creation, naming the rank.

After that point rank 0 is inside ORT's session creation, blocked in `accept()`, and cannot be
told anything. A worker that dies there (out of memory, a mismatched graph) is reported on
stderr by a watchdog:

```
onnxruntime-genai: tensor-parallel rank 3 exited (exit code 1) before joining the collective
group. Rank 0 will stay blocked in session creation. See /tmp/tp/rank3.log.
```

The process will hang; the message tells you which rank to look at. Set `log_dir` so there is
something to look at.
