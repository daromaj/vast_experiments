#!/usr/bin/env python3
"""
Decide whether a rented vast.ai instance is broken, from `show instance --raw`.

    vastai show instance $ID --raw | scripts/instance_fault.py

Prints a one-line reason and exits 0 when the host is unusable; prints NOTHING
when it looks healthy or merely slow. e2e_oneshot.sh treats any output as
"abandon this rental and rent elsewhere" (exit 75).

This lives in its own file for a reason. It used to be a `python3 -c '...'`
heredoc inside e2e_oneshot.sh, and the Python contained single-quoted strings
('unknown'), which silently terminated the SHELL's single-quoted argument. The
interpreter received truncated source and answered SyntaxError on every poll,
so the detector never once ran. Worse, the unit test extracted the source with
a regex and ran it directly, so it never exercised the shell quoting and
reported PASS throughout. A file has no quoting surface and the test now
invokes exactly what production invokes.

It must be wrong in only one direction. A missed fault costs a 20-minute ssh
timeout of billing; a FALSE fault throws away a working host mid-provision,
which is worse and harder to notice. When in doubt, say nothing.
"""
import json
import sys

# Substrings, not exact matches: vast passes the docker daemon message through
# verbatim and it varies by registry, host and failure mode.
#
# Deliberately NOT here: a bare "not found". It is redundant - the real failure
# already matches "failed to resolve reference" and "error response from
# daemon" - and it is broad enough to fire on some benign message that happens
# to contain it.
FATAL = (
    "failed to resolve reference",
    "manifest unknown",
    "pull access denied",
    "error response from daemon",
    "toomanyrequests",
    "no space left",
    "no such host",
    "unauthorized",
    "invalid reference",
)


def verdict(payload):
    """Return a reason string, or '' when nothing is provably wrong."""
    d = payload
    if isinstance(d, list):
        if not d:
            return ""                   # empty list is not evidence of a fault
        d = d[0]
    if not isinstance(d, dict):
        return ""

    msg = (d.get("status_msg") or "").strip()
    status = (d.get("actual_status") or "").lower()

    if any(f in msg.lower() for f in FATAL):
        return f"{status or 'unknown'}: {msg[:200]}"
    if status in ("exited", "offline"):
        return f"{status}: {msg[:200] or 'instance stopped before ssh came up'}"
    return ""


def describe(payload):
    """Everything worth knowing for a post-mortem, fault or not."""
    d = payload
    if isinstance(d, list):
        d = d[0] if d else {}
    if not isinstance(d, dict):
        return "unreadable payload"
    return (f"status={(d.get('actual_status') or '?')!r} "
            f"intent={(d.get('intended_status') or '?')!r} "
            f"msg={((d.get('status_msg') or '').strip()[:300])!r}")


def main():
    try:
        payload = json.load(sys.stdin)
    except Exception:
        # An API hiccup is not a broken host. --describe still has to say
        # something, or the caller's post-mortem line goes blank.
        if "--describe" in sys.argv[1:]:
            print("no parseable status captured")
        return 0
    if "--describe" in sys.argv[1:]:
        print(describe(payload))
        return 0
    line = verdict(payload)
    if line:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
