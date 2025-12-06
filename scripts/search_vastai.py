#!/usr/bin/env python3
"""
Script to search vast.ai offers for both on-demand and interruptible instances.
Combines results, calculates estimated total hourly cost including:

Prerequisite: activate the project virtual environment first, e.g.::

    source .venv/bin/activate
"""
"""
- Base rental cost (dph)
- Storage cost for 70GB container
- Download cost for 70GB of data (one-time, amortized over estimated usage)
Then displays top 15 results sorted by estimated total price.
"""

import subprocess
import json
import sys
from typing import List, Dict

# Search criteria
MIN_GPU_RAM = 24  # GB (minimum for Wan2.1-I2V-14B with offloading)
MIN_DISK_SPACE = 120  # GB (sufficient with cache cleanup)
MAX_INET_COST = 0.001  # $/GB (= $1/TB)
MIN_INET_DOWN_SPEED = 1000  # Mb/s

# GPU filter - exclude incompatible GPUs
# EXCLUDE_GPU_NAMES = ["RTX 5090", "RTX_5090"]  # sm_120 not supported by PyTorch 2.4.1
EXCLUDE_GPU_NAMES = []

# Cost calculation parameters
CONTAINER_SIZE_GB = 120  # GB (sufficient with cache cleanup)
DATA_DOWNLOAD_GB = 100  # GB - downloaded during the 1hr rental period


def run_vastai_search(instance_type: str) -> List[Dict]:
    """
    Run vastai search for given instance type (on-demand or bid).
    Returns list of offers as dictionaries.
    """
    query = (
        f"gpu_ram >= {MIN_GPU_RAM} "
        f"disk_space >= {MIN_DISK_SPACE} "
        f"inet_down_cost < {MAX_INET_COST} "
        f"inet_up_cost < {MAX_INET_COST} "
        f"inet_down >= {MIN_INET_DOWN_SPEED}"
    )
    
    cmd = ["vastai", "search", "offers", query, "--raw"]
    
    # Add type flag
    if instance_type == "bid":
        cmd.append("-b")
    elif instance_type == "on-demand":
        cmd.append("-d")
    
    print(f"Searching {instance_type} instances...", file=sys.stderr)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        offers = json.loads(result.stdout)
        
        # Filter out incompatible GPUs and add instance type
        filtered_offers = []
        for offer in offers:
            gpu_name = offer.get('gpu_name', '')
            # Skip if GPU is in exclusion list
            if any(excluded in gpu_name for excluded in EXCLUDE_GPU_NAMES):
                continue
            offer['instance_type'] = instance_type
            filtered_offers.append(offer)
        
        return filtered_offers
    except subprocess.CalledProcessError as e:
        print(f"Error searching {instance_type}: {e.stderr}", file=sys.stderr)
        return []
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON for {instance_type}: {e}", file=sys.stderr)
        return []


def calculate_total_cost(offer: Dict) -> float:
    """
    Calculate estimated total cost for 1 hour rental including:
    - Base rental cost (dph) for 1 hour
    - Storage cost for container for 1 hour
    - Download cost for 70GB (full one-time cost)
    """
    # Base rental cost per hour
    dph = offer.get('dph_total', offer.get('dph', 0)) or 0
    
    # Storage cost ($/GB/month converted to $/hour)
    # storage_cost is in $/GB/month, convert to hourly
    storage_cost_per_gb_month = offer.get('storage_cost', 0) or 0
    storage_cost_hourly = (CONTAINER_SIZE_GB * storage_cost_per_gb_month) / (30 * 24)
    
    # Download cost (full one-time cost for 70GB during the 1hr rental)
    inet_down_cost = offer.get('inet_down_cost', 0) or 0
    download_cost_total = DATA_DOWNLOAD_GB * inet_down_cost
    
    total_cost = dph + storage_cost_hourly + download_cost_total
    
    return total_cost


def format_table_header() -> str:
    """
    Generate table header for results display.
    """
    header = f"{'#':<4} {'ID':<8} {'Type':<6} {'GPU':<20} {'VRAM':<8} {'Est$/h':<8} {'Base$/h':<8} {'Down':<8} {'Up':<8} {'Loc':<4} {'Rel%':<5} {'TFLOPS':<8}\n"
    header += "-" * 130 + "\n"
    return header


def format_table_row(offer: Dict, rank: int) -> str:
    """
    Format offer as a single table row with critical info only.
    """
    # Extract critical fields
    # Use 'id' (ask_contract_id) which is what vastai create instance needs
    machine_id = str(offer.get('id', offer.get('ask_contract_id', 'N/A')))[:8]
    instance_type = offer.get('instance_type', 'unk')[:6].upper()
    gpu_name = offer.get('gpu_name', 'Unknown').replace('_', ' ')[:20]
    num_gpus = offer.get('num_gpus', 0)
    gpu_ram = offer.get('gpu_ram', 0)
    
    # Format VRAM in human readable way
    if gpu_ram and gpu_ram < 1000:
        vram_str = f"{num_gpus}x{int(gpu_ram)}G"
    else:
        # Fix the ridiculous VRAM values (appears to be in MB not GB)
        vram_gb = gpu_ram / 1024 if gpu_ram > 1000 else gpu_ram
        vram_str = f"{num_gpus}x{int(vram_gb)}G"
    
    # Get TFLOPS (total_flops is in TFLOPS)
    total_flops = offer.get('total_flops', 0)
    tflops_str = f"{total_flops:.1f}" if total_flops else "N/A"
    
    total_cost = offer.get('estimated_total_cost', 0)
    dph = offer.get('dph_total', offer.get('dph', 0)) or 0
    
    inet_down = offer.get('inet_down', 0)
    inet_up = offer.get('inet_up', 0)
    # Convert Mb/s to Gb/s for cleaner display
    down_str = f"{inet_down/1000:.1f}Gb/s" if inet_down >= 1000 else f"{int(inet_down)}Mb/s"
    up_str = f"{inet_up/1000:.1f}Gb/s" if inet_up >= 1000 else f"{int(inet_up)}Mb/s"
    
    geolocation = offer.get('geolocation', 'N/A')[:4]
    reliability = offer.get('reliability', 0) * 100
    
    row = f"{rank:<4} {machine_id:<8} {instance_type:<6} {gpu_name:<20} {vram_str:<8} ${total_cost:<7.4f} ${dph:<7.4f} {down_str:<8} {up_str:<8} {geolocation:<4} {reliability:>4.1f} {tflops_str:<8}\n"
    
    return row


def main():
    """
    Main function to search, combine, calculate, and display results.
    """
    print("Starting vast.ai search...\n", file=sys.stderr)
    
    # Search both types
    on_demand_offers = run_vastai_search("on-demand")
    bid_offers = run_vastai_search("bid")
    
    # Combine results
    all_offers = on_demand_offers + bid_offers
    
    if not all_offers:
        print("No offers found matching criteria!", file=sys.stderr)
        sys.exit(1)
    
    print(f"\nFound {len(all_offers)} total offers ({len(on_demand_offers)} on-demand, {len(bid_offers)} interruptible)\n", file=sys.stderr)
    
    # Calculate total cost for each offer
    for offer in all_offers:
        offer['estimated_total_cost'] = calculate_total_cost(offer)
    
    # Sort by estimated total cost
    all_offers.sort(key=lambda x: x['estimated_total_cost'])
    
    # Display top 15
    top_15 = all_offers[:15]
    
    print("\n" + "="*130)
    print(f"TOP 15 OFFERS (sorted by estimated total hourly cost)")
    print("="*130)
    print(format_table_header(), end='')
    
    for idx, offer in enumerate(top_15, 1):
        print(format_table_row(offer, idx), end='')
    
    print("="*130)
    print(f"\nShowing top 15 of {len(all_offers)} total matching offers")
    print(f"Est$/h = Base rental (1hr) + Storage ({CONTAINER_SIZE_GB}GB for 1hr) + Download cost ({DATA_DOWNLOAD_GB}GB one-time)")
    print(f"\n⚠️  IMPORTANT: For BID instances, MUST use --bid_price when creating instance!")
    print(f"   Example: vastai create instance <ID> --disk 120 --bid_price <DPH+0.01>")
    print(f"\n❌ RTX 5090 filtered out (not compatible with PyTorch 2.4.1)")
    print(f"\n💡 TIP: 120GB is enough if you clean up caches after downloads (see plan step 2.9.2)")


if __name__ == "__main__":
    main()

