#!/usr/bin/env python3
"""
Interactive script to search vast.ai offers for both on-demand and interruptible instances.
Combines results, calculates estimated total hourly cost including:

- Base rental cost (dph)
- Storage cost for 120GB container
- Download cost for 100GB of data (one-time, amortized over estimated usage)
Then displays top 15 results sorted by estimated total price.
Allows interactive selection with arrow keys to bid on instances.

Prerequisite: activate the project virtual environment first, e.g.::

    source .venv/bin/activate
"""
import curses
import subprocess
import json
import sys
from typing import List, Dict, Optional

# Search criteria
MIN_GPU_RAM = 24  # GB (minimum for Wan2.1-I2V-14B with offloading)
MIN_DISK_SPACE = 120  # GB (sufficient with cache cleanup)
MAX_INET_COST = 0.002  # $/GB (= $2/TB)
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
    - Download cost for 100GB (full one-time cost)
    """
    # Base rental cost per hour
    dph = offer.get('dph_total', offer.get('dph', 0)) or 0

    # Storage cost ($/GB/month converted to $/hour)
    # storage_cost is in $/GB/month, convert to hourly
    storage_cost_per_gb_month = offer.get('storage_cost', 0) or 0
    storage_cost_hourly = (CONTAINER_SIZE_GB * storage_cost_per_gb_month) / (30 * 24)

    # Download cost (full one-time cost for 100GB during the 1hr rental)
    inet_down_cost = offer.get('inet_down_cost', 0) or 0
    download_cost_total = DATA_DOWNLOAD_GB * inet_down_cost

    total_cost = dph + storage_cost_hourly + download_cost_total

    return total_cost


def format_table_header() -> str:
    """
    Generate table header for results display.
    """
    header_line = "# ID   Type GPU          VRAM Est$ Base$ Down Up Loc Rel TF\n"
    divider = "-" * len(header_line.rstrip()) + "\n"
    return header_line + divider


def format_table_row(offer: Dict, rank: int, selected: bool = False) -> str:
    """
    Format offer as a single table row with critical info only.
    If selected, highlight with '*' or color if possible.
    """
    prefix = '*' if selected else ' '
    # Extract critical fields
    # Use 'id' (ask_contract_id) which is what vastai create instance needs
    machine_id = str(offer.get('id', offer.get('ask_contract_id', 'N/A')))[:5]
    instance_type = offer.get('instance_type', 'unk')[:3].upper()
    gpu_name = offer.get('gpu_name', 'Unknown').replace('_', ' ')[:12]
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
    down_str = f"{inet_down/1000:.1f}Gb" if inet_down >= 1000 else f"{int(inet_down)}Mb"
    up_str = f"{inet_up/1000:.1f}Gb" if inet_up >= 1000 else f"{int(inet_up)}Mb"

    geolocation = offer.get('geolocation', 'N/A')[:2]
    reliability = offer.get('reliability', 0) * 100

    row = f"{prefix}{rank:<1} {machine_id:<4} {instance_type:<4} {gpu_name:<12} {vram_str:<5} {total_cost:<5.4f} {dph:<5.4f} {down_str:<5} {up_str:<4} {geolocation:<2} {reliability:<3.1f} {tflops_str:<4}\n"

    return row


def create_bid_instance(offer: Dict):
    """
    Create a bid instance for the selected offer.
    """
    machine_id = offer['id']
    dph = offer.get('dph_total', offer.get('dph', 0)) or 0
    # Add 0.01 to bid price as suggested
    bid_price = dph + 0.01

    cmd = ["vastai", "create", "instance", str(machine_id), "--disk", "120", "--bid_price", str(bid_price), "--template_hash", "a3b79706f4f5ed8164bb1fadaeea2718"]

    print(f"\n🖥️ Creating bid instance for machine {machine_id} with bid price ${bid_price:.2f}...")
    try:
        result = subprocess.run(cmd, check=True, text=True)
        print(f"✅ Bid instance created successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create bid instance: {e.stderr}")
        return False


def curses_interactive_select(offers: List[Dict]) -> Optional[int]:
    """
    Use curses to display interactive menu for selecting an offer.
    Returns the index of the selected offer, or None if quit.
    """

    def _draw_menu(stdscr, offers, selected_row_idx):
        stdscr.clear()

        # Init color pair for highlighting
        curses.start_color()
        curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)  # Highlight selected row

        # Get table width
        header_line_example = "# ID   Type GPU          VRAM Est$ Base$ Down Up Loc Rel TF\n"
        header_width = len(header_line_example.rstrip())

        # Prepare lines
        lines = [f"TOP {len(offers)} OFFERS (sorted by estimated total hourly cost) - Selected: #{selected_row_idx+1}"]
        lines.append("=" * header_width)

        # Header
        header_lines = format_table_header().split('\n')
        for line in header_lines:
            if line.strip():
                lines.append(line)

        # Rows
        for idx, offer in enumerate(offers):
            row = format_table_row(offer, idx+1).rstrip('\n')
            lines.append(row)

        lines.append("=" * header_width)
        lines.append("")
        lines.append(f"Est$/h = Base rental (1hr) + Storage ({CONTAINER_SIZE_GB}GB for 1hr) + Download cost ({DATA_DOWNLOAD_GB}GB one-time)")
        lines.append("")
        lines.append("WARNING: For BID instances, MUST use --bid_price when creating instance!")
        lines.append("   Example: vastai create instance <ID> --disk 120 --bid_price <DPH+0.01>")
        lines.append("")
        lines.append("Filtered out: RTX 5090 (not compatible with PyTorch 2.4.1)")
        lines.append("")
        lines.append("TIP: 120GB is enough if you clean up caches after downloads (see plan step 2.9.2)")
        lines.append("")
        lines.append("Use UP/DOWN arrows to navigate, ENTER to bid, Q to quit")

        # Draw lines
        max_y, max_x = stdscr.getmaxyx()
        for i, line in enumerate(lines):
            if i < max_y:
                truncated = line[:max_x-1]
                try:
                    if i >= len(lines) - 10:  # Instructions at bottom
                        stdscr.addstr(i, 0, truncated)
                    else:
                        row_idx = i - len(lines) + 10  # Calculate actual row index for offers
                        if i >= 4 and i < 4 + len(offers):
                            row_idx = i - 4
                            if row_idx == selected_row_idx:
                                stdscr.attron(curses.color_pair(1))
                                stdscr.addstr(i, 0, truncated)
                                stdscr.attroff(curses.color_pair(1))
                            else:
                                stdscr.addstr(i, 0, truncated)
                        else:
                            stdscr.addstr(i, 0, truncated)
                except curses.error:
                    pass

        stdscr.refresh()

    def _select_loop(stdscr, offers):
        curses.curs_set(0)
        current_row = 0
        _draw_menu(stdscr, offers, current_row)
        while True:
            key = stdscr.getch()
            if key == curses.KEY_UP and current_row > 0:
                current_row -= 1
            elif key == curses.KEY_DOWN and current_row < len(offers) - 1:
                current_row += 1
            elif key in [10, 13, curses.KEY_ENTER]:
                break
            elif key in [ord('q'), ord('Q')]:
                return None
            _draw_menu(stdscr, offers, current_row)
        return current_row

    return curses.wrapper(_select_loop, offers)


def main():
    """
    Main function to search, combine, calculate, display, and allow interactive selection.
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

    # Display top 15 (or fewer if less available)
    top_offers = all_offers[:15]

    if not top_offers:
        return

    # Start interactive selection
    selected_idx = curses_interactive_select(top_offers)

    if selected_idx is None:
        print("Selection cancelled.")
        curses.endwin()  # Just in case
        return

    # Bid on selected
    selected_offer = top_offers[selected_idx]
    print(f"Selected instance: ID {selected_offer['id']}")
    confirm = input("Are you sure you want to bid on this instance? (y/n): ").strip().lower()
    if confirm == 'y':
        create_bid_instance(selected_offer)
    else:
        print("Bid cancelled.")


if __name__ == "__main__":
    main()
