#!/usr/bin/env python3
"""
Script to search vast.ai offers focusing on NVIDIA H100/H200 GPUs.

Prerequisite: activate the project virtual environment first, e.g.::

    source .venv/bin/activate

The script queries both on-demand and interruptible offers, filters down to
H100/H200 accelerators, calculates an estimated hourly cost (base rental,
storage, and download amortization), and prints a ranked summary table.
"""

import json
import subprocess
import sys
from typing import Dict, List

# Search criteria tuned for H100/H200 workloads
MIN_GPU_RAM = 80  # GB
MIN_DISK_SPACE = 160  # GB
MAX_INET_COST = 0.001  # $/GB (= $1/TB)
MIN_INET_DOWN_SPEED = 1000  # Mb/s

# GPU filters
TARGET_GPU_KEYWORDS = ["H100", "H200"]
EXCLUDE_GPU_NAMES = ["RTX 5090", "RTX_5090"]

# Cost calculation parameters (scaled up from the generic search)
CONTAINER_SIZE_GB = 160
DATA_DOWNLOAD_GB = 150


def gpu_matches_target(gpu_name: str) -> bool:
    """Return True when the GPU name references either H100 or H200."""
    normalized = (gpu_name or "").upper()
    return any(keyword in normalized for keyword in TARGET_GPU_KEYWORDS)


def run_vastai_search(instance_type: str) -> List[Dict]:
    """Invoke `vastai search offers` for the requested instance type."""
    query = (
        f"gpu_ram >= {MIN_GPU_RAM} "
        f"disk_space >= {MIN_DISK_SPACE} "
        f"inet_down_cost < {MAX_INET_COST} "
        f"inet_up_cost < {MAX_INET_COST} "
        f"inet_down >= {MIN_INET_DOWN_SPEED}"
    )

    cmd = ["vastai", "search", "offers", query, "--raw"]

    if instance_type == "bid":
        cmd.append("-b")
    elif instance_type == "on-demand":
        cmd.append("-d")

    print(f"Searching {instance_type} instances...", file=sys.stderr)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        offers = json.loads(result.stdout)

        filtered_offers = []
        for offer in offers:
            gpu_name = offer.get("gpu_name", "")
            if any(excluded in gpu_name for excluded in EXCLUDE_GPU_NAMES):
                continue
            if not gpu_matches_target(gpu_name):
                continue
            offer["instance_type"] = instance_type
            filtered_offers.append(offer)

        return filtered_offers
    except subprocess.CalledProcessError as exc:
        print(f"Error searching {instance_type}: {exc.stderr}", file=sys.stderr)
        return []
    except json.JSONDecodeError as exc:
        print(f"Error parsing JSON for {instance_type}: {exc}", file=sys.stderr)
        return []


def calculate_total_cost(offer: Dict) -> float:
    """Compute estimated hourly cost including storage and download amortization."""
    dph = offer.get("dph_total", offer.get("dph", 0)) or 0
    storage_cost_per_gb_month = offer.get("storage_cost", 0) or 0
    storage_cost_hourly = (CONTAINER_SIZE_GB * storage_cost_per_gb_month) / (30 * 24)
    inet_down_cost = offer.get("inet_down_cost", 0) or 0
    download_cost_total = DATA_DOWNLOAD_GB * inet_down_cost
    return dph + storage_cost_hourly + download_cost_total


def format_table_header() -> str:
    header = (
        f"{'#':<4} {'ID':<8} {'Type':<6} {'GPU':<20} {'VRAM':<8} "
        f"{'Est$/h':<8} {'Base$/h':<8} {'Down':<8} {'Up':<8} {'Loc':<4} "
        f"{'Rel%':<5} {'TFLOPS':<8}\n"
    )
    header += "-" * 130 + "\n"
    return header


def format_table_row(offer: Dict, rank: int) -> str:
    machine_id = str(offer.get("id", offer.get("ask_contract_id", "N/A")))[:8]
    instance_type = offer.get("instance_type", "unk")[:6].upper()
    gpu_name = offer.get("gpu_name", "Unknown").replace("_", " ")[:20]
    num_gpus = offer.get("num_gpus", 0)
    gpu_ram = offer.get("gpu_ram", 0)

    if gpu_ram and gpu_ram < 1000:
        vram_str = f"{num_gpus}x{int(gpu_ram)}G"
    else:
        vram_gb = gpu_ram / 1024 if gpu_ram > 1000 else gpu_ram
        vram_str = f"{num_gpus}x{int(vram_gb)}G"

    total_flops = offer.get("total_flops", 0)
    tflops_str = f"{total_flops:.1f}" if total_flops else "N/A"

    total_cost = offer.get("estimated_total_cost", 0)
    dph = offer.get("dph_total", offer.get("dph", 0)) or 0

    inet_down = offer.get("inet_down", 0)
    inet_up = offer.get("inet_up", 0)
    down_str = f"{inet_down/1000:.1f}Gb/s" if inet_down >= 1000 else f"{int(inet_down)}Mb/s"
    up_str = f"{inet_up/1000:.1f}Gb/s" if inet_up >= 1000 else f"{int(inet_up)}Mb/s"

    geolocation = offer.get("geolocation", "N/A")[:4]
    reliability = offer.get("reliability", 0) * 100

    return (
        f"{rank:<4} {machine_id:<8} {instance_type:<6} {gpu_name:<20} {vram_str:<8} "
        f"${total_cost:<7.4f} ${dph:<7.4f} {down_str:<8} {up_str:<8} {geolocation:<4} "
        f"{reliability:>4.1f} {tflops_str:<8}\n"
    )


def main() -> None:
    print("Starting vast.ai search for H100/H200 instances...\n", file=sys.stderr)

    on_demand_offers = run_vastai_search("on-demand")
    bid_offers = run_vastai_search("bid")
    all_offers = on_demand_offers + bid_offers

    if not all_offers:
        print("No H100/H200 offers found matching criteria!", file=sys.stderr)
        sys.exit(1)

    print(
        f"\nFound {len(all_offers)} total offers "
        f"({len(on_demand_offers)} on-demand, {len(bid_offers)} interruptible)\n",
        file=sys.stderr,
    )

    for offer in all_offers:
        offer["estimated_total_cost"] = calculate_total_cost(offer)

    all_offers.sort(key=lambda offer: offer["estimated_total_cost"])

    top_30 = all_offers[:30]

    print("\n" + "=" * 130)
    print("TOP 30 H100/H200 OFFERS (sorted by estimated total hourly cost)")
    print("=" * 130)
    print(format_table_header(), end="")

    for idx, offer in enumerate(top_30, 1):
        print(format_table_row(offer, idx), end="")

    print("=" * 130)
    print(f"\nShowing top 30 of {len(all_offers)} total matching offers")
    print(
        f"Est$/h = Base rental (1hr) + Storage ({CONTAINER_SIZE_GB}GB for 1hr) + "
        f"Download cost ({DATA_DOWNLOAD_GB}GB one-time)",
    )
    print(
        "\n⚠️  IMPORTANT: For BID instances, MUST use --bid_price when creating instance!",
    )
    print("   Example: vastai create instance <ID> --disk 160 --bid_price <DPH+0.01>")
    print("\n❌ RTX 5090 filtered out (not compatible with PyTorch 2.4.1)")


if __name__ == "__main__":
    main()
