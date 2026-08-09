# Setup

Repo root is `C:\ISS`.

## Option A — native Windows (start here)

Everything works except the Triton kernel, which the engine detects and falls
back from automatically.

```powershell
cd C:\ISS
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Install a CUDA build of PyTorch, not the CPU default:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Check the 4060 is visible:

```powershell
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.is_available())"
```

## Option B — WSL2 (needed only for the Triton kernel)

Phase 2's fused attention kernel needs Triton, which is unreliable on native
Windows. `C:\ISS` appears inside WSL as `/mnt/c/ISS`.

```bash
cd /mnt/c/ISS
python3 -m venv ~/.venvs/iss          # keep the venv on the Linux side
source ~/.venvs/iss/bin/activate
pip install -r requirements.txt
pip install triton
nvidia-smi
```

The `/mnt/c` bridge is slow for many small files, so `pip install` and
`git status` feel sluggish. It does not affect benchmark numbers — the hot path
is GPU compute, not disk.

Verify the kernel matches the reference before trusting any number from it:

```bash
pytest tests/test_paged_attention.py -q -k triton
```

## Model download

The first run pulls Qwen2.5-0.5B-Instruct (~1 GB) from HuggingFace. Keep it off
the C: partition:

```powershell
setx HF_HOME "D:\hf_cache"            # Windows, then reopen the shell
```
```bash
export HF_HOME=/mnt/d/hf_cache        # WSL
```

## Tests

The suite needs no GPU and downloads nothing — it builds a tiny randomly
initialised Qwen2 in memory.

```bash
pytest -q                             # everything
pytest tests/test_block_manager.py tests/test_scheduler.py -q   # no torch needed
```

## Git

```bash
cd C:\ISS
git init
git add .
git commit -m "phase 0-5: paged KV cache, continuous batching, quantisation"
```

Commit per phase from here on. A history that walks from naive baseline to
paged attention to quantisation is itself a strong signal to whoever reads the
repo.

## Running order

```bash
python -m bench.run_baseline --mode static --batch-sizes 1,2,4,8,16,32 --n 32
python -m bench.run_engine   --n 64 --workload skewed
python -m bench.sweep
python -m bench.plot
```

## Troubleshooting

**CUDA out of memory during load.** Lower `--gpu-util` (default 0.85). The
profiler sizes the KV cache from whatever is free after the weights and one
worst-case forward pass, so a lower value just means fewer blocks.

**`request N needs X KV blocks but the cache holds Y`.** A single request is
larger than the whole cache. Reduce `--output-len`, or raise `--gpu-util`.

**`out of KV blocks ... with preemption disabled`.** Expected — that is what
`--no-preemption` demonstrates.

**Throughput lower than the static baseline at batch size 1.** Also expected.
Continuous batching wins on aggregate throughput under load, not on a single
request; scheduling has per-step Python overhead that a single sequence pays
without getting anything back.
