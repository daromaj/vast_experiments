#!/usr/bin/env python3
"""
Search vast.ai for LongCat-Video Avatar-1.5 compatible GPUs.

Filters for:
- RTX 5090 / RTX 4090 (best speed/price for DiT inference)
- PCIe >= 20 GB/s (critical for Cache-DIT CPU offload + multi-GPU CP)
- CUDA >= 12.4 (required by torch 2.6.0+cu124)
- VRAM >= 24 GB (minimum for INT8 480p), prefers 32 GB+
- Clean internet: download cost < $1/GB, decent speed
- Disk >= 150 GB (~45 GB models + container + outputs)

Prerequisite: activate the project virtual environment first, e.g.:

    source .venv/bin/activate

Usage:
    python scripts/search_longcat.py              # on-demand + bid, all GPUs
    python scripts/search_longcat.py --gpu 5090   # RTX 5090 only
    python scripts/search_longcat.py --gpu 4090   # RTX 4090 only
    python scripts/search_longcat.py --multi      # multi-GPU only (CP-capable)
"""

import subprocess
import json
import sys
import argparse
from typing import List, Dict, Optional

# ==============================================================================
# Config — tuned for LongCat-Video Avatar-1.5
# ==============================================================================
MIN_GPU_RAM = 24            # GB — 480p INT8 fits, 32 GB preferred for 720p
MIN_DISK_SPACE = 150        # GB — ~45 GB models + venv + outputs
MIN_INET_DOWN = 500         # Mb/s — 45 GB in ~12 min at this speed
MAX_INET_COST = 1.0         # $/GB — flag anything above as scam
MIN_PCIE_BW = 15            # GB/s — below this, CPU offload & CP bottleneck hard
# CUDA filter applied post-search (cuda_max_good field doesn't work in vastai query)
MIN_CUDA = 12.4             # torch 2.6.0+cu124 requirement
MIN_RELIABILITY = 0.97      # skip flaky hosts

# Template: vastai/pytorch:cuda-12.9.1-auto
# Vast.ai auto-selects latest PyTorch matching the GPU's CUDA.
# 5090 (CUDA 13.x) → should get torch 2.11+ required by Cache-DIT.
# LongCat-Video pins torch==2.6.0 — provision script overrides to >=2.6.0.
TEMPLATE_HASH = "8ab860114bddc24c4cb43af37aacfa15"
TEMPLATE_DESC = "vastai/pytorch (default, cuda auto-detect)"

CONTAINER_SIZE_GB = 150     # disk reservation
DATA_DOWNLOAD_GB = 50       # ~45 GB models + a bit of margin

# GPU bandwidth rankings (GB/s) — for reference in output
GPU_BANDWIDTH = {
    "RTX_5090": 1792,
    "RTX_4090": 1008,
    "RTX_5080": 960,
    "RTX_4080": 717,
    "L40S": 864,
    "A6000": 768,
    "A100": 2039,
    "H100": 3352,
}


def build_query(args) -> str:
    """Build vastai search query from filters."""
    parts = [
        f"gpu_ram >= {MIN_GPU_RAM}",
        f"disk_space >= {MIN_DISK_SPACE}",
        f"inet_down >= {MIN_INET_DOWN}",
        f"inet_down_cost < {MAX_INET_COST}",
        f"inet_up_cost < {MAX_INET_COST}",
        f"pcie_bw >= {MIN_PCIE_BW}",
        f"reliability > {MIN_RELIABILITY}",
    ]

    if args.gpu:
        gpu_map = {"5090": "RTX_5090", "4090": "RTX_4090", "5080": "RTX_5080"}
        gpu_name = gpu_map.get(args.gpu, args.gpu)
        parts.append(f"gpu_name={gpu_name}")

    if args.multi:
        parts.append("num_gpus >= 2")

    return " ".join(parts)


def run_search(instance_type: str, query: str) -> List[Dict]:
    """Run vastai search and return parsed offers."""
    cmd = ["vastai", "search", "offers", query, "--raw"]
    if instance_type == "bid":
        cmd.append("-b")
    elif instance_type == "on-demand":
        cmd.append("-d")

    print(f"  Searching {instance_type}...", file=sys.stderr)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        offers = json.loads(result.stdout)
        for o in offers:
            o["instance_type"] = instance_type

        # CUDA filter (post-search — cuda_max_good not supported in vastai query syntax)
        filtered = []
        for o in offers:
            cuda = o.get("cuda_max_good")
            if cuda is None or cuda >= MIN_CUDA:
                filtered.append(o)
        return filtered

    except Exception as e:
        print(f"  Error: {e}", file=sys.stderr)
        return []


def calc_cost(offer: Dict) -> float:
    """Total estimated hourly cost: rental + storage + amortized download."""
    dph = offer.get("dph_total", offer.get("dph", 0)) or 0
    storage_cost_per_gb_month = offer.get("storage_cost", 0) or 0
    storage_hourly = (CONTAINER_SIZE_GB * storage_cost_per_gb_month) / (30 * 24)
    download_total = DATA_DOWNLOAD_GB * (offer.get("inet_down_cost", 0) or 0)
    return dph + storage_hourly + download_total


def fmt_vram(vram_mb: int, num_gpus: int) -> str:
    """VRAM from API is in MB for values > 1000, GB otherwise."""
    if vram_mb > 1000:
        gb = vram_mb / 1024
    else:
        gb = float(vram_mb)
    total = gb * num_gpus
    if num_gpus > 1:
        return f"{num_gpus}×{gb:.0f}G"
    return f"{gb:.0f}G"


def fmt_membw(bw_gb_per_sec: float, num_gpus: int) -> str:
    """GPU memory bandwidth — API returns GB/s per GPU."""
    total = bw_gb_per_sec * num_gpus
    if total >= 1000:
        return f"{total/1000:.1f}TB/s"
    return f"{total:.0f}GB/s"


def print_table(offers: List[Dict]):
    """Pretty-print results table with key metrics for DiT inference."""
    hdr = (
        f"{'#':<3} {'ID':<9} {'Type':<5} {'GPU':<16} {'N':<2} "
        f"{'VRAM':<6} {'MemBW':<8} {'PCIe':<6} {'$/h':<7} "
        f"{'DLP/$':<7} {'↓Inet':<7} {'↑Inet':<7} {'Loc':<4} {'Rel':<5} {'Days':<6}"
    )
    sep = "─" * len(hdr)

    print(f"\n{sep}")
    print(hdr)
    print(sep)

    for i, o in enumerate(offers, 1):
        oid = str(o.get("id", "?"))[:8]
        itype = o.get("instance_type", "?")[:3]
        gpu = o.get("gpu_name", "?").replace("_", " ")[:16]
        num = o.get("num_gpus", 1)
        vram = fmt_vram(o.get("gpu_ram", 0), num)
        membw = fmt_membw(o.get("gpu_mem_bw", 0), num)
        pcie = f"PCIe{o.get('pci_gen',0):.0f}x{o.get('gpu_lanes', 0)}" if o.get("gpu_lanes") else f"{o.get('pcie_bw', 0):.0f}GB/s"
        cost = o.get("estimated_total_cost", 0)
        dlp = o.get("dlperf_per_dphtotal", 0)
        down = f"{o.get('inet_down', 0)/1000:.1f}G"
        up = f"{o.get('inet_up', 0)/1000:.1f}G"
        loc = (o.get("geolocation", "?") or "?")[:4]
        rel = f"{o.get('reliability', 0)*100:.1f}%"
        dur_h = o.get("duration", 0) or 0
        days = f"{dur_h/24:.0f}d" if dur_h > 0 else "?"

        star = " ★" if dlp > 300 else ""

        print(
            f"{i:<3} {oid:<9} {itype:<5} {gpu:<16} {num:<2} "
            f"{vram:<6} {membw:<8} {pcie:<6} ${cost:<6.4f} "
            f"{dlp:<7.0f} {down:<7} {up:<7} {loc:<4} {rel:<5} {days:<6}{star}"
        )

    print(sep)


def main():
    parser = argparse.ArgumentParser(
        description="Search vast.ai for LongCat-Video compatible GPUs"
    )
    parser.add_argument("--gpu", help="GPU filter: 5090, 4090, 4080, or full name")
    parser.add_argument("--multi", action="store_true", help="Multi-GPU only (for context parallelism)")
    parser.add_argument("--limit", type=int, default=20, help="Max results to show")
    parser.add_argument("--sort", default="cost", choices=["cost", "dlp"], help="Sort by cost or DLP/$")
    parser.add_argument("--type", default="on-demand", choices=["on-demand", "bid", "all"],
                        help="Instance type: on-demand (default), bid, or all")
    args = parser.parse_args()

    query = build_query(args)

    print(f"🔍 LongCat-Video GPU search", file=sys.stderr)
    print(f"   Query: {query}", file=sys.stderr)
    print(f"   Model: ~45 GB download, INT8 DiT + Cache-DIT offload", file=sys.stderr)
    print(f"   Target: 480p (24 GB VRAM min), 720p (32 GB VRAM preferred)", file=sys.stderr)
    print(file=sys.stderr)

    # Search requested types
    all_offers = []
    if args.type in ("on-demand", "all"):
        all_offers += run_search("on-demand", query)
    if args.type in ("bid", "all"):
        all_offers += run_search("bid", query)

    if not all_offers:
        print("\n❌ No offers found! Try relaxing filters.", file=sys.stderr)
        sys.exit(1)

    # Calculate costs
    for o in all_offers:
        o["estimated_total_cost"] = calc_cost(o)

    # Sort
    if args.sort == "dlp":
        all_offers.sort(key=lambda x: -(x.get("dlperf_per_dphtotal", 0)))
    else:
        all_offers.sort(key=lambda x: x["estimated_total_cost"])

    # Display
    top = all_offers[: args.limit]
    print_table(top)

    # Summary
    on_d = sum(1 for o in top if o["instance_type"] == "on-demand")
    bid_n = sum(1 for o in top if o["instance_type"] == "bid")
    best_5090 = [o for o in top if "5090" in o.get("gpu_name", "")]
    best_4090 = [o for o in top if "4090" in o.get("gpu_name", "")]

    print(f"\n📊 {len(all_offers)} total matches, showing top {len(top)} ({on_d} on-demand, {bid_n} bid)")
    if best_5090:
        o = best_5090[0]
        print(f"🏆 Best 5090: ID {o['id']} — ${o['estimated_total_cost']:.3f}/h, DLP/$ {o.get('dlperf_per_dphtotal',0):.0f}, {o.get('geolocation','?')}")
    if best_4090:
        o = best_4090[0]
        print(f"🥈 Best 4090: ID {o['id']} — ${o['estimated_total_cost']:.3f}/h, DLP/$ {o.get('dlperf_per_dphtotal',0):.0f}, {o.get('geolocation','?')}")
    print(f"\n💡 DLP/$ > 300 = very good speed/price for DiT inference")
    print(f"💡 PCIe gen shown as PCIe<gen>x<lanes> — prefer 4x16 or 5x16 for CPU offload")
    tmpl = TEMPLATE_HASH
    print(f"💡 To create: vastai create instance <ID> --disk 150 --template_hash {tmpl}")
    print(f"   Template: {TEMPLATE_DESC}")
    print(f"   Provisioning overrides torch pin for Cache-DIT >=2.11 compatibility")

if __name__ == "__main__":
    main()
