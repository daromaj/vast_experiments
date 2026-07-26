#!/usr/bin/env python3
"""
Apply the measured quick wins to the SHIPPED (GUI-format) workflows.

scripts/apply_quick_wins.py does this for API-format JSON, where every input is
addressed by name. The workflows a human actually loads in ComfyUI are the GUI
format: values live in a positional `widgets_values` list and non-widget inputs
are edges in a `links` array. Same three changes, different surgery.

Measured on an RTX 5090, 8s clip, run 2 of 2: 87.7s -> 81.7s.

R1  use_disk_cache -> true on WanVideoTextEncodeCached
    Confirmed working from the log ("Loading prompt embeds from cache: ...").
    umt5-xxl (~11GB) is not loaded at all on a prompt hit.

R2  wire WanVideoTorchCompileSettings -> WanVideoVAELoader.compile_args
    Compiles vae.model.decoder, which is invoked once per latent frame in a
    Python loop. This is a real edge, so it needs a new entry in `links`, the
    source node's output `links` list, and the destination node's input `link`.

R5  force_offload -> false on WanVideoSampler
    Post-run only, so it costs nothing inside a run, but it pushes ~16GB to CPU
    and the next queued generation pays to load it back.

Widget indices are asserted against their expected current values rather than
trusted, because widgets_values is positional: if a node gains an input in a
future WanVideoWrapper release every index after it shifts, and a silent
mis-write would corrupt the workflow in a way that only shows up as bad output.
"""
import argparse
import json

T_TEXT = "WanVideoTextEncodeCached"
T_VAE = "WanVideoVAELoader"
T_SAMPLER = "WanVideoSampler"
T_COMPILE = "WanVideoTorchCompileSettings"

# type -> (widget index, expected current value, new value, label)
WIDGET_FIXES = {
    T_TEXT: (5, False, True, "R1 use_disk_cache -> true"),
    T_SAMPLER: (5, True, False, "R5 force_offload -> false"),
}


def node_by_type(wf, t):
    found = [n for n in wf["nodes"] if n.get("type") == t]
    if len(found) != 1:
        raise SystemExit(f"expected exactly 1 {t}, found {len(found)}")
    return found[0]


def apply_widget_fixes(wf):
    changed = []
    for t, (idx, expected, new, label) in WIDGET_FIXES.items():
        node = node_by_type(wf, t)
        vals = node["widgets_values"]
        if vals[idx] == new:
            continue
        if vals[idx] != expected:
            raise SystemExit(
                f"{t} widget {idx} is {vals[idx]!r}, expected {expected!r} - "
                "widget order likely changed upstream, refusing to write")
        vals[idx] = new
        changed.append(f"{label} (node {node['id']})")
    return changed


def wire_vae_compile(wf):
    """Connect the compile settings node to the VAE loader's compile_args input."""
    vae = node_by_type(wf, T_VAE)
    comp = node_by_type(wf, T_COMPILE)

    slot = next((i for i in vae.get("inputs", [])
                 if i.get("name") == "compile_args"), None)
    if slot is None:
        raise SystemExit(f"{T_VAE} has no compile_args input")
    if slot.get("link") is not None:
        return []

    link_id = int(wf.get("last_link_id") or 0) + 1
    # [id, src_node, src_slot, dst_node, dst_slot, type]
    wf["links"].append([link_id, comp["id"], 0, vae["id"], 0, "WANCOMPILEARGS"])
    slot["link"] = link_id

    out = comp["outputs"][0]
    out["links"] = (out.get("links") or []) + [link_id]
    wf["last_link_id"] = link_id

    return [f"R2 compile_args: {comp['id']} -> {vae['id']} (link {link_id})"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()

    for path in args.files:
        wf = json.load(open(path))
        changed = apply_widget_fixes(wf) + wire_vae_compile(wf)
        if changed:
            with open(path, "w") as fh:
                json.dump(wf, fh, indent=2)
        print(f"{path}: {len(changed)} change(s)")
        for line in changed:
            print(f"    {line}")


if __name__ == "__main__":
    raise SystemExit(main())
