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

## Option B — WSL2 (needed for the Triton kernel)

The fused paged-attention kernel needs Triton, which does not support native
Windows. Everything else runs fine without it — the engine detects Triton and
falls back to the torch backend. `C:\ISS` appears inside WSL as `/mnt/c/ISS`,
so there is nothing to copy.

**1. Install WSL2** (PowerShell as Administrator, then reboot):

```powershell
wsl --install
```

Already have it? `wsl --update` and skip ahead. After reboot, Ubuntu opens and
asks for a username and password — any values, it is a local account.

**2. Check the GPU is visible from Linux.** Recent NVIDIA drivers expose the
GPU to WSL automatically; there is no CUDA driver to install inside Ubuntu.

```bash
nvidia-smi
```

If that fails, update your Windows NVIDIA driver rather than installing
anything in Ubuntu.

**3. Environment.** Keep the venv on the Linux filesystem (`~`), not on
`/mnt/c` — the bridge is slow for many small files. Benchmark numbers are
unaffected either way, since the hot path is GPU compute, not disk.

```bash
sudo apt update && sudo apt install -y python3-venv python3-pip
python3 -m venv ~/iss-venv
source ~/iss-venv/bin/activate
pip install torch                      # Linux PyPI torch IS the CUDA build,
                                       # and it pulls in Triton automatically
pip install transformers accelerate numpy matplotlib pytest
```

**4. Reuse the model you already downloaded** instead of pulling another 1 GB:

```bash
export HF_HOME=/mnt/d/hf_cache
echo 'export HF_HOME=/mnt/d/hf_cache' >> ~/.bashrc
```

**5. Verify Triton is there:**

```bash
cd /mnt/c/ISS
python -c "import triton, torch; print(triton.__version__, torch.cuda.get_device_name(0))"
```

**6. Correctness before speed.** Never report a number from a kernel you have
not validated:

```bash
python -m pytest tests/test_paged_attention.py -v -k triton
```

Six cases, checking the kernel against the torch backend on randomly permuted
block tables and ragged sequence lengths.

Use `python -m pytest`, not bare `pytest`. Ubuntu ships a system pytest on
`PATH` that shadows the venv's, and it reads `/usr/lib/python3/dist-packages`
instead of your virtualenv -- so it reports `No module named 'numpy'` for
packages you just installed. If the header says `plugins: typeguard-...`, you
are running the system one. `python -m` always uses the interpreter you
activated.

**7. Benchmark it:**

```bash
python -m bench.bench_attention     # kernel in isolation, speedup vs context
pytest -q                           # full suite, now with Triton
python -m bench.diagnose            # same probe as Windows -> WDDM comparison
python -m bench.sweep --quick       # backend experiment now has both rows
python -m bench.plot
```

`bench_attention` is the one that matters for the report: it isolates the
attention op from the ~1300 other kernels in a step, so you can see what the
kernel itself did, and it sweeps context length because that is the axis the
fused kernel should win on.

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
