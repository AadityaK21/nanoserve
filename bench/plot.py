"""Turn results/sweep.json into the figures for the report.

    python -m bench.plot

Writes PNGs to results/. Missing experiments are skipped rather than fatal, so
this works on a partial sweep.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"


def _load():
    p = RESULTS / "sweep.json"
    if not p.exists():
        raise SystemExit("no results/sweep.json -- run `python -m bench.sweep` first")
    return json.loads(p.read_text())


def _ok(d) -> bool:
    return isinstance(d, dict) and "error" not in d


def _save(fig, name: str) -> None:
    fig.tight_layout()
    out = RESULTS / name
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def plot_throughput(r):
    static = r.get("static_baseline")
    cont = r.get("continuous")
    if not _ok(static) or not _ok(cont):
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, wl in zip(axes, ("skewed", "uniform")):
        if wl not in static:
            continue
        pts = sorted(
            ((int(k[2:]), v["output_tok_per_s"]) for k, v in static[wl].items()),
            key=lambda x: x[0],
        )
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", label="static batching")
        if wl in cont:
            ax.axhline(cont[wl]["output_tok_per_s"], color="crimson", ls="--",
                       label="continuous batching")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("batch size")
        ax.set_ylabel("output tokens/s")
        ax.set_title(f"{wl} lengths")
        ax.grid(alpha=0.3)
        ax.legend()
    _save(fig, "fig1_throughput.png")


def plot_latency_vs_load(r):
    d = r.get("latency_vs_load")
    if not _ok(d):
        return
    rates, ttft, tpot, e2e = [], [], [], []
    for k, v in sorted(d.items(), key=lambda kv: float(kv[0][4:])):
        rates.append(float(k[4:]))
        ttft.append(v["ttft_p99_ms"])
        tpot.append(v["tpot_p99_ms"])
        e2e.append(v["e2e_p99_ms"])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, ys, label in zip(axes, (ttft, tpot, e2e), ("p99 TTFT (ms)", "p99 TPOT (ms)", "p99 E2E (ms)")):
        ax.plot(rates, ys, "o-")
        ax.set_xlabel("arrival rate (req/s)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    fig.suptitle("Tail latency vs offered load")
    _save(fig, "fig2_latency_vs_load.png")


def plot_chunked(r):
    d = r.get("chunked_prefill")
    if not _ok(d):
        return
    keys = list(d)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, metric, label in zip(axes, ("ttft_p99_ms", "tpot_p99_ms"),
                                 ("p99 TTFT (ms)", "p99 TPOT (ms)")):
        vals = [d[k].get(metric, 0) for k in keys]
        colors = ["#3b7dd8" if k.startswith("chunked") else "#d8733b" for k in keys]
        ax.bar(range(len(keys)), vals, color=colors)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(keys, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Chunked prefill: TTFT cost vs TPOT benefit")
    _save(fig, "fig3_chunked_prefill.png")


def plot_block_size(r):
    d = r.get("block_size")
    if not _ok(d):
        return
    keys = sorted(d, key=lambda k: int(k[5:]))
    sizes = [int(k[5:]) for k in keys]
    fig, ax1 = plt.subplots(figsize=(6.5, 4.2))
    ax1.plot(sizes, [d[k]["output_tok_per_s"] for k in keys], "o-", color="#3b7dd8")
    ax1.set_xlabel("block size (tokens)")
    ax1.set_ylabel("output tokens/s", color="#3b7dd8")
    ax1.set_xscale("log", base=2)
    ax2 = ax1.twinx()
    ax2.plot(sizes, [d[k].get("kv_capacity_tokens", 0) for k in keys], "s--", color="#d8733b")
    ax2.set_ylabel("KV capacity (tokens)", color="#d8733b")
    ax1.grid(alpha=0.3)
    ax1.set_title("Paging granularity")
    _save(fig, "fig4_block_size.png")


def plot_quant(r):
    d = r.get("quantization")
    if not _ok(d):
        return
    keys = [k for k in ("None", "int8", "int4") if k in d and _ok(d[k])]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].bar(keys, [d[k]["engine"]["weight_mib"] for k in keys], color="#3b7dd8")
    axes[0].set_ylabel("weights (MiB)")
    axes[1].bar(keys, [d[k]["engine"]["kv_capacity_tokens"] for k in keys], color="#d8733b")
    axes[1].set_ylabel("KV capacity (tokens)")
    axes[2].bar(keys, [d[k]["output_tok_per_s"] for k in keys], color="#4aa564")
    axes[2].set_ylabel("output tokens/s")
    for ax in axes:
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Weight-only quantisation: memory freed becomes KV capacity")
    _save(fig, "fig5_quantization.png")


def plot_backend(r):
    d = r.get("backend")
    if not _ok(d):
        return
    keys = [k for k in d if _ok(d[k])]
    if len(keys) < 2:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].bar(keys, [d[k]["output_tok_per_s"] for k in keys], color="#3b7dd8")
    axes[0].set_ylabel("output tokens/s")
    axes[1].bar(keys, [d[k]["tpot_p50_ms"] for k in keys], color="#d8733b")
    axes[1].set_ylabel("p50 TPOT (ms)")
    for ax in axes:
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Attention backend: torch gather vs fused Triton kernel")
    _save(fig, "fig6_backend.png")


def main() -> None:
    r = _load()
    for fn in (plot_throughput, plot_latency_vs_load, plot_chunked,
               plot_block_size, plot_quant, plot_backend):
        try:
            fn(r)
        except Exception as exc:  # pragma: no cover
            print(f"!! {fn.__name__}: {exc}")


if __name__ == "__main__":
    main()
