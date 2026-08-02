#!/usr/bin/env python3
"""
Pin the host-fault detector. No network, no rental.

The detector decides whether to abandon a paid rental, so it has to be wrong in
only one direction. A missed fault costs a 20-minute ssh timeout of billing; a
FALSE fault throws away a perfectly good host mid-provision, which is worse and
harder to notice. The healthy cases below matter more than the faulty ones.

HOW THIS IS TESTED, AND WHY IT CHANGED
--------------------------------------
The previous version of this test regex-extracted the Python out of a
`python3 -c '...'` heredoc in e2e_oneshot.sh and ran it via subprocess with no
shell involved. It reported PASS on every case while the real detector was
100% broken: the embedded Python contained single-quoted strings, those quotes
closed the SHELL's single-quoted argument early, and python3 got truncated
source. Machine 140178 burned a full 22-minute rental on 2026-08-02 emitting
`SyntaxError: '{' was never closed` once per poll, undetected.

So this test now runs BASH over the real function text from e2e_oneshot.sh,
with `vastai` stubbed on PATH. Whatever quoting the shell does to the command,
this test does too. A test that skips the layer where the bug lives is worse
than no test - it converts "untested" into "verified", which is a lie.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
# Overridable so this test can be pointed at an OLD revision of the script to
# confirm it actually FAILS there. A regression test nobody has seen fail is
# just a passing test.
SCRIPT = os.environ.get("E2E_SCRIPT") or os.path.join(HERE, "e2e_oneshot.sh")

# The real message, verbatim from `vastai show instance 46605392` on 2026-08-02.
REAL = ("Error response from daemon: failed to resolve reference "
        '"docker.io/vastai/comfy:v0.28.0-cuda-12.9-py312": '
        "docker.io/vastai/comfy:v0.28.0-cuda-12.9-py312: not found")

CASES = [
    # (name, status_msg, actual_status, expect_fault)
    ("the actual Hong Kong failure", REAL, "loading", True),
    ("manifest unknown", "manifest unknown: manifest tagged v9 not found",
     "loading", True),
    ("docker hub rate limit",
     "toomanyrequests: You have reached your pull rate limit", "loading", True),
    ("disk full", "write /var/lib/docker: no space left on device",
     "loading", True),
    ("private registry auth", "pull access denied for vastai/comfy",
     "loading", True),
    ("instance died before ssh", "", "exited", True),

    # --- healthy: these must NOT trip it ------------------------------------
    ("still pulling the image", "", "loading", False),
    ("pulling, with progress text", "Downloading image layers 4/9",
     "loading", False),
    ("up and running", "", "running", False),
    ("running with a benign message", "Container created successfully",
     "running", False),
    ("empty payload", None, None, False),
    ("benign mention of the image",
     "Pulling vastai/comfy:v0.28.0-cuda-12.9-py312", "loading", False),
    # Quoting hazards, in the data this time. The message is host-supplied and
    # arrives verbatim, so it will eventually contain every awkward character.
    ("apostrophe in a benign message",
     "host's disk is warming up", "loading", False),
    ("apostrophe in a fatal message",
     "Error response from daemon: host's registry is unreachable",
     "loading", True),
    ("shell metacharacters in a benign message",
     "progress $(whoami) `id` \"quoted\" 'single' 100%", "loading", False),
]


def extract_function():
    """Take the real function text so bash parses it exactly as in production."""
    src = open(SCRIPT).read()
    m = re.search(r"^instance_fault\(\) \{\n(.*?)^\}\n", src, re.S | re.M)
    if not m:
        sys.exit(f"could not find instance_fault() in {SCRIPT} - did it move?")
    return m.group(0)


def run(func_text, payload, tmp):
    """Run the bash function with `vastai` stubbed to emit `payload`."""
    bindir = os.path.join(tmp, "bin")
    os.makedirs(bindir, exist_ok=True)
    stub = os.path.join(bindir, "vastai")
    blob = os.path.join(tmp, "payload.json")
    with open(blob, "w") as fh:
        fh.write("<<UNPARSEABLE>>" if payload is Ellipsis
                 else json.dumps(payload))
    with open(stub, "w") as fh:
        fh.write(f'#!/usr/bin/env bash\ncat {blob}\n')
    os.chmod(stub, 0o755)

    rundir = os.path.join(tmp, "run")
    os.makedirs(rundir, exist_ok=True)
    env = dict(os.environ, PATH=bindir + os.pathsep + os.environ["PATH"])
    script = (f'set -uo pipefail\nREPO={HERE!r}/..\nINSTANCE=1\n'
              f'RUN_DIR={rundir!r}\n{func_text}\ninstance_fault\n')
    p = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                       env=env, timeout=60)
    err = os.path.join(rundir, "fault_probe.err")
    stderr = open(err).read() if os.path.exists(err) else ""
    return p.stdout.strip(), (p.stderr + stderr).strip()


def main():
    func_text = extract_function()
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, msg, status, expect in CASES:
            payload = {}
            if msg is not None:
                payload["status_msg"] = msg
            if status is not None:
                payload["actual_status"] = status
            got, err = run(func_text, [payload] if payload else [], tmp)
            ok = bool(got) == expect and not err
            print(f"{'PASS' if ok else 'FAIL'}  {name}")
            if not ok:
                print(f"      expected {'a fault' if expect else 'no fault'}, "
                      f"got {got!r}")
                if err:
                    print(f"      stderr: {err[:300]}")
                failures.append(name)

        # Malformed input must be treated as "no fault": an API hiccup is not
        # evidence that the host is broken, and destroying on it would be a bug.
        got, err = run(func_text, Ellipsis, tmp)
        ok = not got and not err
        print(f"{'PASS' if ok else 'FAIL'}  unparseable API response is not a "
              f"fault")
        if not ok:
            print(f"      got {got!r} stderr {err[:300]}")
            failures.append("unparseable API response")

        # The bug that made all of the above meaningless was an interpreter
        # error, not a wrong verdict. Assert explicitly that the classifier
        # actually RUNS, so a broken probe can never masquerade as "healthy".
        got, err = run(func_text, [{"status_msg": REAL,
                                    "actual_status": "loading"}], tmp)
        ok = "failed to resolve reference" in got and "Error" not in err
        print(f"{'PASS' if ok else 'FAIL'}  classifier executes (no interpreter "
              f"error on the real payload)")
        if not ok:
            print(f"      got {got!r} stderr {err[:300]}")
            failures.append("classifier executes")

    print()
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        return 1
    print(f"all {len(CASES) + 2} fault-detector checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
