#!/usr/bin/env python3
"""
Pin the host-fault detector in e2e_oneshot.sh. No network, no rental.

The detector decides whether to abandon a paid rental, so it has to be wrong in
only one direction. A missed fault costs a 20-minute ssh timeout of billing; a
FALSE fault throws away a perfectly good host mid-provision, which is worse and
harder to notice. The healthy cases below matter more than the faulty ones.

The classifier is extracted from the shell script by parsing it out, so this
test cannot drift away from what actually runs.
"""
import json
import os
import re
import subprocess
import sys

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "e2e_oneshot.sh")

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
    # The word "loading" alone is not a fault, and neither is a message that
    # merely mentions an image name.
    ("benign mention of the image",
     "Pulling vastai/comfy:v0.28.0-cuda-12.9-py312", "loading", False),
]


def extract_classifier():
    """Pull the python heredoc out of instance_fault() so it cannot drift."""
    src = open(SCRIPT).read()
    m = re.search(r"instance_fault\(\)\s*\{.*?python3 -c '(.*?)'\n", src,
                  re.S)
    if not m:
        sys.exit("could not find the instance_fault classifier in "
                 f"{SCRIPT} - did it move?")
    return m.group(1)


def run(code, payload):
    p = subprocess.run([sys.executable, "-c", code],
                       input=json.dumps(payload), capture_output=True,
                       text=True)
    if p.returncode != 0:
        return f"<crashed: {p.stderr.strip()[:120]}>"
    return p.stdout.strip()


def main():
    code = extract_classifier()
    failures = []
    for name, msg, status, expect in CASES:
        payload = {}
        if msg is not None:
            payload["status_msg"] = msg
        if status is not None:
            payload["actual_status"] = status
        got = run(code, [payload] if payload else [])
        ok = bool(got) == expect and not got.startswith("<crashed")
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
        if not ok:
            print(f"      expected {'a fault' if expect else 'no fault'}, "
                  f"got {got!r}")
            failures.append(name)

    # Malformed input must be treated as "no fault": an API hiccup is not
    # evidence that the host is broken, and destroying on it would be a bug.
    p = subprocess.run([sys.executable, "-c", code], input="not json",
                       capture_output=True, text=True)
    ok = p.returncode == 0 and not p.stdout.strip()
    print(f"{'PASS' if ok else 'FAIL'}  unparseable API response is not a fault")
    if not ok:
        failures.append("unparseable API response")

    print()
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        return 1
    print(f"all {len(CASES) + 1} fault-detector checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
