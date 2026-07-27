#!/usr/bin/env python3
"""
Build the three compile arms for a one-shot 58 s render.

    scripts/build_compile_ab.py --out workflows/generated/compile_ab

The sweep never measured compile-off. Both variants named "nocomp"
(s3_4step_nocomp, s6_sage_nocomp) only switch `mode` from
`max-autotune-no-cudagraphs` to `default` — every workflow in the repo still
wires `WanVideoTorchCompileSettings` into `WanVideoModelLoader.compile_args`. So
"is compile worth it on a single clip?" has been argued from numbers that never
tested it.

Three arms, all from the same s5-lineage base so nothing else varies:

  a_autotune  max-autotune-no-cudagraphs   (what ships today)
  b_default   mode=default                 (compiles, skips autotune)
  c_nocompile compile_args removed         (never measured before)

Each is run once on a cold cache, which is the one-shot case. Arms do not
contaminate each other: inductor caches per mode, and arm C compiles nothing.
"""
import argparse
import copy
import json
import os

BASE = "workflows/generated/full/full58_sage_API.json"
LOADER = "122"          # WanVideoModelLoader
COMPILE_NODE = "177"    # WanVideoTorchCompileSettings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--out", default="workflows/generated/compile_ab")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    base = json.load(open(args.base))

    loader = base.get(LOADER)
    if not loader or "compile_args" not in loader.get("inputs", {}):
        raise SystemExit(f"{args.base}: node {LOADER} has no compile_args")
    if base.get(COMPILE_NODE, {}).get("class_type") != "WanVideoTorchCompileSettings":
        raise SystemExit(f"{args.base}: node {COMPILE_NODE} is not the compile settings node")

    written = []

    a = copy.deepcopy(base)
    a[COMPILE_NODE]["inputs"]["mode"] = "max-autotune-no-cudagraphs"
    written.append(("a_autotune", a))

    b = copy.deepcopy(base)
    b[COMPILE_NODE]["inputs"]["mode"] = "default"
    written.append(("b_default", b))

    c = copy.deepcopy(base)
    del c[LOADER]["inputs"]["compile_args"]
    # Drop the settings node too. Left in place it is an orphan that ComfyUI
    # would still instantiate, and an orphan named "TorchCompileSettings" in a
    # workflow labelled no-compile is exactly the sort of thing that gets
    # misread later.
    refs = [k for k, v in c.items()
            if isinstance(v, dict)
            for val in (v.get("inputs") or {}).values()
            if isinstance(val, list) and val and val[0] == COMPILE_NODE]
    if refs:
        raise SystemExit(f"node {COMPILE_NODE} still referenced by {refs}")
    del c[COMPILE_NODE]
    written.append(("c_nocompile", c))

    for name, wf in written:
        path = os.path.join(args.out, f"{name}_API.json")
        json.dump(wf, open(path, "w"), indent=2)
        n122 = wf[LOADER]["inputs"]
        mode = wf.get(COMPILE_NODE, {}).get("inputs", {}).get("mode", "-")
        print(f"{path}  compile_args={'compile_args' in n122}  mode={mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
