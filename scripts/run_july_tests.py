#!/usr/bin/env python3
"""
Execute the july_test.md matrix against a running ComfyUI and record wall-clock
timings. Runs ON the instance (stdlib only, no extra deps).

Each workflow runs twice. The FIRST run is discarded: torch.compile spends
minutes compiling on first use per shape, and max-autotune far more, so only the
second run is a meaningful generation time. This is the whole reason the test
procedure exists rather than just timing one run.

Results are written incrementally to --out so a crash, disconnect or timeout
still leaves everything measured up to that point.

    python3 run_july_tests.py --workflows a.json b.json --runs 2 --out results.json
"""
import argparse
import json
import time
import urllib.error
import urllib.request
import uuid

BASE = "http://127.0.0.1:18188"


def api(path, payload=None, timeout=30):
    url = f"{BASE}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"} if data else {}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def wait_for_comfy(limit=300):
    """ComfyUI may still be starting; do not mistake that for a failure."""
    start = time.time()
    while time.time() - start < limit:
        try:
            api("/system_stats")
            return True
        except Exception:
            time.sleep(5)
    return False


def bump_seeds(workflow, run):
    """
    Give every seed-bearing node a fresh value.

    ComfyUI caches execution on node inputs, so re-queueing a byte-identical
    workflow returns the previous result without running anything: observed as a
    3.0s "generation" that emitted the same output filename as the 171.5s run
    before it. Varying the seed forces real resampling while leaving the
    torch.compile cache warm, which is exactly what run 2 is supposed to measure.
    """
    changed = []
    for nid, node in workflow.items():
        inputs = node.get("inputs") or {}
        for key in ("seed", "noise_seed"):
            if isinstance(inputs.get(key), int):
                inputs[key] = (inputs[key] + run * 1000 + 7) % (2**31)
                changed.append(f"{nid}.{key}")
    return changed


def run_once(workflow, client_id, run_timeout):
    """Queue one generation, block until it leaves the queue, return timing."""
    t0 = time.time()
    resp = api("/prompt", {"prompt": workflow, "client_id": client_id})
    pid = resp["prompt_id"]

    while True:
        elapsed = time.time() - t0
        if elapsed > run_timeout:
            return {"prompt_id": pid, "seconds": elapsed, "status": "TIMEOUT"}

        hist = api(f"/history/{pid}")
        if pid in hist:
            entry = hist[pid]
            st = entry.get("status") or {}
            status = st.get("status_str", "unknown")

            # ComfyUI reports execution failures as status_str="error" with the
            # real cause buried in status.messages. Without pulling it out, a
            # failed run says only "error" and the GPU time is wasted twice.
            node_error = None
            for m in st.get("messages", []) or []:
                if isinstance(m, (list, tuple)) and len(m) == 2 and m[0] == "execution_error":
                    d = m[1] or {}
                    node_error = {
                        "node_id": d.get("node_id"),
                        "node_type": d.get("node_type"),
                        "exception_type": d.get("exception_type"),
                        "exception_message": str(d.get("exception_message"))[:1000],
                    }
                    break

            outputs = []
            for node_out in (entry.get("outputs") or {}).values():
                for key in ("gifs", "images", "videos"):
                    for item in node_out.get(key, []) or []:
                        if isinstance(item, dict) and item.get("filename"):
                            outputs.append(item["filename"])
            return {
                "prompt_id": pid,
                "seconds": time.time() - t0,
                "status": status,
                "outputs": outputs,
                "node_error": node_error,
            }
        time.sleep(3)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workflows", nargs="+", required=True)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--out", default="july_results.json")
    ap.add_argument("--run-timeout", type=float, default=2400, help="seconds per generation")
    args = ap.parse_args()

    if not wait_for_comfy():
        print("FATAL: ComfyUI never became reachable", flush=True)
        return 1

    client_id = str(uuid.uuid4())
    results = []

    for path in args.workflows:
        name = path.split("/")[-1].replace("_API.json", "")
        workflow = json.load(open(path))
        for run in range(1, args.runs + 1):
            label = "compile/warmup (discarded)" if run == 1 else "MEASURED"
            print(f"\n=== {name} run {run}/{args.runs} — {label} ===", flush=True)
            t = time.time()
            seeded = bump_seeds(workflow, run)
            print(f"    reseeded: {', '.join(seeded) or 'NONE - results may be cached!'}",
                  flush=True)
            try:
                r = run_once(workflow, client_id, args.run_timeout)
            except urllib.error.HTTPError as e:
                body = e.read().decode()[:800]
                r = {"status": f"HTTP {e.code}", "error": body, "seconds": time.time() - t}
            except Exception as e:  # noqa: BLE001 - a failed variant must not kill the matrix
                r = {"status": "ERROR", "error": repr(e), "seconds": time.time() - t}

            r.update({"workflow": name, "run": run, "measured": run > 1})
            results.append(r)
            print(
                f"    status={r['status']} seconds={r['seconds']:.1f} "
                f"({r['seconds']/60:.1f}m) outputs={r.get('outputs')}",
                flush=True,
            )
            if r.get("error"):
                print(f"    error: {r['error'][:400]}", flush=True)
            if r.get("node_error"):
                ne = r["node_error"]
                print(f"    FAILED node {ne['node_id']} [{ne['node_type']}] "
                      f"{ne['exception_type']}: {ne['exception_message'][:300]}", flush=True)

            # Write after every run: a later failure must not lose earlier numbers.
            with open(args.out, "w") as fh:
                json.dump(results, fh, indent=2)

            # A variant that cannot run once will not run twice either.
            if run == 1 and r["status"] not in ("success",):
                print("    first run did not succeed — skipping remaining runs "
                      "for this workflow", flush=True)
                break

    print("\n================ SUMMARY ================", flush=True)
    for r in results:
        if r.get("measured"):
            print(f"  {r['workflow']:<24} {r['seconds']/60:6.2f} min   {r['status']}", flush=True)
    print("ALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
