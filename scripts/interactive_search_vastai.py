#!/usr/bin/env python3
"""
Interactive script to search vast.ai offers for both on-demand and interruptible instances.
Combines results, calculates estimated total hourly cost including:

- Base rental cost (dph)
- Storage cost for the container disk (CONTAINER_SIZE_GB)
- Download cost for the model payload (DATA_DOWNLOAD_GB, one-time)
Then displays top 15 results sorted by estimated total price.
Allows interactive selection with arrow keys to bid on instances.

Prerequisite: activate the project virtual environment first, e.g.::

    source .venv/bin/activate
"""
import curses
import re
import shlex
import subprocess
import json
import sys
import urllib.request
from typing import List, Dict, Optional

# Search criteria
MIN_GPU_RAM = 24  # GB (4090=24GB floor; 5090=32GB. query is in GB, raw JSON is MB)
MIN_DISK_SPACE = 60  # GB. Measured on a live 5090 rental 2026-07-26, not guessed:
# 32GB models + 8.8GB venv + 1.6GB inductor cache + 0.6GB nodes/SageAttention = ~43GB fixed.
# A generated video costs ~16MB (ComfyUI writes a silent .mp4 AND an -audio.mp4), so 10 60s
# clips add ~160MB - video count is irrelevant at this scale, the models dominate. 60 leaves
# ~17GB of headroom. Raise to 80 if you also provision the 720P checkpoint (+17GB).
MIN_INET_DOWN_SPEED = 5000  # Mb/s floor. Download speed is the real bottleneck, and a hard
# floor (not the soft cost term below) is what guarantees fast hosts. 5 Gbps costs ~nothing on
# price - the cheapest 5090s often already have 7+ Gbps. Drop to 2000-3000 if the pool looks thin.
MAX_DPH = 0.60  # $/hr hard cap on rental price - very fast hosts get pricey, this reins it in
MIN_PCIE_BW = 20  # GB/s floor on measured PCIe link bandwidth. PCIe 4.0 x16 is ~31.5 GB/s
# theoretical (~20-26 measured); PCIe 3.0 x16 and 4.0 x8 both cap at ~15.75. A 20 floor keeps
# real 4.0 x16 hosts and drops the old/crippled slots (where V100/3090-era boards live).

# Cost calculation parameters
CONTAINER_SIZE_GB = 60  # GB (matches MIN_DISK_SPACE / the --disk value used on create)
DATA_DOWNLOAD_GB = 34  # GB - actual model payload after dropping the unused fp8 encoder
MAX_DOWNLOAD_COST = 1.0  # USD cap on total bandwidth for the payload (not a per-GB rate)
# Per-GB bandwidth price cap derived from the total-cost target above.
MAX_INET_COST = MAX_DOWNLOAD_COST / DATA_DOWNLOAD_GB  # $/GB

# GPU filter - allowlist of card families we actually want.
# Focus: RTX 4090/5090 "or better" == modern Ada/Blackwell cards with NATIVE fp8 and
# >=4090-class compute. This is a substring match against gpu_name, so "L40" also
# catches "L40S", "RTX 4090" catches "RTX 4090D", etc.
# Deliberately excluded: A100 (Ampere, no native fp8), RTX 3090/A40 (Ampere), Tesla V100
# (Volta), RTX PRO 4000 (below 4090 tier). Empty list => allow everything.
# NOTE: RTX 5090 (Blackwell/sm_120) is the PRIMARY target - SageAttention is built for the
# detected host arch in povision_fp8.sh.
INCLUDE_GPU_NAMES = [
    "RTX 4090",
    "RTX 5090",
    "L40",          # L40 / L40S - Ada datacenter, fp8, ~4090-class
    "RTX 6000Ada",
    "RTX 5880Ada",
    "RTX PRO 6000",  # Blackwell workstation flagship
]

# GPU filter - exclude incompatible GPUs (applied after the allowlist, for one-off blocks)
EXCLUDE_GPU_NAMES = []

# --- Instance configuration -------------------------------------------------
#
# Deliberately NOT a saved template. Everything below is passed explicitly on the
# create command so the full instance config lives in git and is reviewable in a
# diff, instead of sitting in Vast's web UI where it drifts out from under us.
# A saved template hash would silently override nothing and hide everything.
#
# Image tag is PINNED, not one of the floating `cuda-12.9-auto` tags. A floating
# tag moves torch between rentals, which ABI-invalidates any cached SageAttention
# wheel at random - the likeliest reason the prebuilt wheel path got written off
# as unreliable. Pinned tag == cacheable wheel. py312 matches the cp312 wheels
# in python/.
#
# Bumping this tag invalidates any harvested SageAttention wheel: the wheel is
# keyed to the torch ABI this specific image ships. Harvest a fresh one after any
# bump (povision_fp8.sh prints the ABI-keyed filename on a source build).
#
# check_for_newer_image() below reports newer tags at startup, so this does not
# have to be checked by hand.
IMAGE_REPO = "vastai/comfy"
IMAGE_VARIANT = "cuda-12.9-py312"  # the flavour we pin within; py312 matches the cp312 wheels
IMAGE_VERSION = "v0.28.0"
IMAGE = f"{IMAGE_REPO}:{IMAGE_VERSION}-{IMAGE_VARIANT}"

PROVISIONING_SCRIPT = (
    "https://raw.githubusercontent.com/daromaj/vast_experiments/"
    "refs/heads/master/povision_fp8.sh"
)

# --use-sage-attention makes ComfyUI hard-depend on SageAttention being importable.
# If provisioning ends with "SAGE: UNAVAILABLE" in the log, expect ComfyUI to fail
# here rather than quietly falling back.
COMFYUI_ARGS = (
    "--disable-auto-launch --port 18188 --enable-cors-header "
    "--disable-xformers --use-sage-attention"
)

PORTS = [1111, 8080, 8384, 72299, 8188, 8288]

PORTAL_CONFIG = (
    "localhost:1111:11111:/:Instance Portal|"
    "localhost:8188:18188:/:ComfyUI|"
    "localhost:8288:18288:/docs:API Wrapper|"
    "localhost:8188:18188:/:ComfyUI|"
    "localhost:8080:18080:/:Jupyter|"
    "localhost:8080:8080:/terminals/1:Jupyter Terminal|"
    "localhost:8384:18384:/:Syncthing"
)


def build_env_string() -> str:
    """
    Assemble the -p/-e blob that `vastai create instance --env` expects.

    It is one opaque string to the CLI, so it is built from parts here rather than
    kept as a single unreadable literal - a typo inside it does not fail loudly,
    it just produces an instance that is subtly wrong.
    """
    parts = [f"-p {p}:{p}" for p in PORTS]
    parts += [
        '-e COMFYUI_VERSION=latest',
        f'-e COMFYUI_ARGS="{COMFYUI_ARGS}"',
        '-e COMFYUI_API_BASE=http://localhost:18188',
        f'-e PROVISIONING_SCRIPT={PROVISIONING_SCRIPT}',
        f'-e PORTAL_CONFIG="{PORTAL_CONFIG}"',
        '-e OPEN_BUTTON_PORT=1111',
        '-e JUPYTER_DIR=/',
        '-e DATA_DIRECTORY=/workspace/',
        '-e OPEN_BUTTON_TOKEN=1',
    ]
    return " ".join(parts)


def _parse_version(tag: str) -> Optional[tuple]:
    """
    'v0.28.0-cuda-12.9-py312' -> (0, 28, 0). None if it is not our variant.

    Numeric tuples, never string comparison: 'v0.3.60' sorts ABOVE 'v0.28.0'
    lexically, which would recommend downgrading to a 2025 image.
    """
    m = re.match(rf"^v(\d+)\.(\d+)\.(\d+)-{re.escape(IMAGE_VARIANT)}$", tag)
    return tuple(int(g) for g in m.groups()) if m else None


def check_for_newer_image() -> None:
    """
    Report newer vastai/comfy tags in our variant. Advisory only - never blocks
    or fails the search, since this runs right before spending money and a Docker
    Hub hiccup is no reason to stop.
    """
    url = (
        f"https://hub.docker.com/v2/repositories/{IMAGE_REPO}/tags"
        f"?page_size=100&name={IMAGE_VARIANT}"
    )
    try:
        with urllib.request.urlopen(url, timeout=8) as resp:
            results = json.load(resp).get("results", [])
    except Exception as e:  # noqa: BLE001 - advisory check, never fatal
        print(f"(image version check skipped: {e})", file=sys.stderr)
        return

    current = _parse_version(f"{IMAGE_VERSION}-{IMAGE_VARIANT}")
    tags = [(v, t["name"], (t.get("last_updated") or "")[:10])
            for t in results if (v := _parse_version(t["name"]))]
    if not tags or current is None:
        return

    newest_ver, newest_tag, newest_date = max(tags)
    if newest_ver > current:
        newer = sorted((t for t in tags if t[0] > current), reverse=True)
        print(f"\n  NEWER IMAGE AVAILABLE: {newest_tag} ({newest_date})", file=sys.stderr)
        print(f"  pinned: {IMAGE_VERSION}-{IMAGE_VARIANT}"
              f"  ({len(newer)} newer release(s))", file=sys.stderr)
        print(f"  To bump: set IMAGE_VERSION = \"v{'.'.join(map(str, newest_ver))}\"",
              file=sys.stderr)
        print("  Note: a bump invalidates any harvested SageAttention wheel "
              "(torch ABI changes).\n", file=sys.stderr)
    else:
        print(f"Image {IMAGE_VERSION}-{IMAGE_VARIANT} is current "
              f"(latest: {newest_tag}, {newest_date}).", file=sys.stderr)


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
        f"inet_down >= {MIN_INET_DOWN_SPEED} "
        f"pcie_bw >= {MIN_PCIE_BW} "
        f"dph_total <= {MAX_DPH}"
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
            # Keep only allowlisted GPU families (empty list => allow everything)
            if INCLUDE_GPU_NAMES and not any(inc in gpu_name for inc in INCLUDE_GPU_NAMES):
                continue
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

    # Download cost (full one-time transfer charge for the model payload)
    inet_down_cost = offer.get('inet_down_cost', 0) or 0
    download_cost_total = DATA_DOWNLOAD_GB * inet_down_cost

    # Rental burned WHILE the payload downloads. This is the real reason a slow host is
    # expensive: you pay dph the whole time it's pulling models. Converts speed -> $ so a
    # cheaper-but-slower host is weighed fairly against a pricey-but-fast one.
    inet_down_mbps = offer.get('inet_down', 0) or 0
    if inet_down_mbps > 0:
        download_seconds = (DATA_DOWNLOAD_GB * 8 * 1000) / inet_down_mbps  # GB->gigabit->megabit / Mbps
    else:
        download_seconds = 0
    offer['download_minutes'] = download_seconds / 60
    wasted_rental = dph * (download_seconds / 3600)

    total_cost = dph + storage_cost_hourly + download_cost_total + wasted_rental

    return total_cost


# Single source of truth for the results table layout: (label, width, align).
# Header and every row are rendered from this one spec, so they always line up
# and each cell is truncated to its column width (no drift, no overflow).
_COLUMNS = [
    ("#",     3, '<'),
    ("ID",    9, '<'),
    ("Type",  4, '<'),
    ("GPU",  12, '<'),
    ("VRAM",  5, '<'),
    ("Est$",  6, '>'),
    ("Base$", 6, '>'),
    ("Down",  6, '>'),
    ("DLm",   4, '>'),
    ("Up",    6, '>'),
    ("Loc",   3, '<'),
    ("Rel",   5, '>'),
    ("TF",    6, '>'),
]
_COL_SEP = " "


def _render_row(cells: List[str]) -> str:
    """Join pre-computed cell strings using the shared _COLUMNS widths/alignment."""
    return _COL_SEP.join(
        f"{str(cell)[:width]:{align}{width}}"
        for cell, (_label, width, align) in zip(cells, _COLUMNS)
    )


def table_width() -> int:
    """Total rendered width of the table (for dividers / box-drawing)."""
    return sum(w for _l, w, _a in _COLUMNS) + len(_COL_SEP) * (len(_COLUMNS) - 1)


def format_table_header() -> str:
    """
    Generate table header for results display.
    """
    header = _render_row([label for label, _w, _a in _COLUMNS])
    divider = "-" * len(header)
    return header + "\n" + divider + "\n"


def format_table_row(offer: Dict, rank: int, selected: bool = False) -> str:
    """
    Format offer as a single table row with critical info only.
    If selected, highlight with '*' or color if possible.
    """
    prefix = '*' if selected else ' '
    # Extract critical fields
    # Use 'id' (ask_contract_id) which is what vastai create instance needs
    machine_id = str(offer.get('id', offer.get('ask_contract_id', 'N/A')))
    instance_type = offer.get('instance_type', 'unk')[:3].upper()
    gpu_name = offer.get('gpu_name', 'Unknown').replace('_', ' ')
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

    dl_str = f"{offer.get('download_minutes', 0):.0f}m"

    cells = [
        f"{prefix}{rank}",
        machine_id,
        instance_type,
        gpu_name,
        vram_str,
        f"{total_cost:.4f}",
        f"{dph:.4f}",
        down_str,
        dl_str,
        up_str,
        geolocation,
        f"{reliability:.1f}",
        tflops_str,
    ]

    return _render_row(cells) + "\n"


def create_instance(offer: Dict):
    """
    Create an instance for the selected offer.

    Branches on the offer's instance_type: 'bid' (interruptible) requires --bid_price,
    'on-demand' must NOT pass it (passing a bid price on an on-demand offer makes it
    interruptible). Previously this always bid regardless of the selected type.
    """
    machine_id = offer['id']
    instance_type = offer.get('instance_type', 'bid')
    dph = offer.get('dph_total', offer.get('dph', 0)) or 0

    cmd = ["vastai", "create", "instance", str(machine_id),
           "--image", IMAGE,
           "--env", build_env_string(),
           "--onstart-cmd", "entrypoint.sh",
           "--disk", str(CONTAINER_SIZE_GB),
           "--jupyter", "--ssh", "--direct"]

    if instance_type == "bid":
        bid_price = dph + 0.01  # bid slightly above the shown price
        cmd += ["--bid_price", str(bid_price)]
        print(f"\n🖥️ Creating BID (interruptible) instance {machine_id} at bid ${bid_price:.3f}/hr...")
    else:
        print(f"\n🖥️ Creating ON-DEMAND instance {machine_id} at ${dph:.3f}/hr...")

    # Echo the exact command. --env is a single opaque blob that the CLI will not
    # validate, so a mistake in it produces a running, billing, subtly-wrong box.
    # Seeing it before it executes is the only cheap check available.
    print("\n" + shlex.join(cmd) + "\n")

    try:
        subprocess.run(cmd, check=True, text=True)
        print(f"✅ Instance created successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create instance (exit {e.returncode}): {e}")
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

        # Get table width from the shared column spec (header + rows use the same)
        header_width = table_width()

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
        lines.append(f"Est$/h = rental(1hr) + storage({CONTAINER_SIZE_GB}GB/1hr) + download charge({DATA_DOWNLOAD_GB}GB) + rental burned during download")
        lines.append("")
        gpu_filter = ", ".join(INCLUDE_GPU_NAMES) if INCLUDE_GPU_NAMES else "any"
        for fline in [
            "Applied filters (all must hold):",
            f"  GPU allowlist : {gpu_filter}",
            f"  gpu_ram       >= {MIN_GPU_RAM} GB",
            f"  disk_space    >= {MIN_DISK_SPACE} GB",
            f"  pcie_bw       >= {MIN_PCIE_BW} GB/s (PCIe 4.0 x16)",
            f"  inet_down     >= {MIN_INET_DOWN_SPEED} Mb/s",
            f"  inet_dn/up$   <  ${MAX_INET_COST:.4f}/GB (<=${MAX_DOWNLOAD_COST} for {DATA_DOWNLOAD_GB}GB)",
            f"  dph_total     <= ${MAX_DPH}/hr",
        ]:
            lines.append(fline)
        lines.append("")
        lines.append(f"Down = host inet_down. DLm = est. minutes to pull {DATA_DOWNLOAD_GB}GB (the real time sink on slow hosts).")
        lines.append("")
        lines.append("BID offers create interruptible (auto --bid_price); on-demand offers create fixed-price.")
        lines.append("")
        lines.append("Use UP/DOWN arrows to navigate, ENTER to select, Q to quit")

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

    # Before spending money: is the pinned image still the newest one?
    check_for_newer_image()

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

    # Create the selected instance (bid or on-demand per its type)
    selected_offer = top_offers[selected_idx]
    itype = selected_offer.get('instance_type', 'bid')
    dph = selected_offer.get('dph_total', selected_offer.get('dph', 0)) or 0
    print(f"Selected: ID {selected_offer['id']}  type={itype}  ~${dph:.3f}/hr")
    confirm = input(f"Create this {itype} instance? (y/n): ").strip().lower()
    if confirm == 'y':
        create_instance(selected_offer)
    else:
        print("Cancelled.")


if __name__ == "__main__":
    main()
