#!/usr/bin/env bash
# Run e2e_oneshot.sh, and move to another host when one cannot start at all.
#
#   ./scripts/e2e_oneshot_retry.sh            # up to 3 hosts
#   ATTEMPTS=5 ./scripts/e2e_oneshot_retry.sh
#
# A rented host that cannot pull the container image never opens ssh. Vast puts
# the reason in status_msg, e2e_oneshot.sh now reads it and exits 75 instead of
# waiting out a 20-minute timeout, and this loop rents somewhere else. Machine
# 139787 (Hong Kong) hit exactly this on 2026-08-02: "failed to resolve
# reference ... not found" for an image tag that was present on Docker Hub, so
# a broken registry mirror on the host.
#
# The retry is deliberately narrow. Only exit 75 - "this host is unusable" - is
# retried. A failed RENDER is not retried: that is a real problem worth looking
# at, and spending another rental to see it again helps nobody.
set -uo pipefail
trap '' PIPE

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

ATTEMPTS="${ATTEMPTS:-3}"
SKIP=""
RUN_ROOT="${RUN_ROOT:-$REPO/output}"

say() { echo "[retry $(date -u +%H:%M:%S)] $*"; }

for attempt in $(seq 1 "$ATTEMPTS"); do
    run_dir="$RUN_ROOT/e2e_$(date -u +%Y%m%dT%H%M%SZ)_a${attempt}"
    say "attempt ${attempt}/${ATTEMPTS}${SKIP:+ (skipping${SKIP})} -> $run_dir"

    SKIP="$SKIP" RUN_DIR="$run_dir" "$REPO/scripts/e2e_oneshot.sh" best
    rc=$?

    if [[ $rc -eq 0 ]]; then
        say "done -> $run_dir"
        echo "$run_dir" > "$RUN_ROOT/.last_successful_run"
        exit 0
    fi

    if [[ $rc -ne 75 ]]; then
        say "e2e_oneshot exited $rc - not a host fault, not retrying"
        exit "$rc"
    fi

    # Learn which machine to avoid. search_cheap_egress logs the line it rented
    # from, so the id comes from what actually happened rather than a guess.
    bad=$(grep -oE "^machine [0-9]+" "$run_dir/create.log" 2>/dev/null \
          | head -1 | awk '{print $2}')
    if [[ -n "$bad" ]]; then
        SKIP="$SKIP $bad"
        say "host fault on machine $bad: $(cat "$run_dir/host_fault.txt" 2>/dev/null | head -c 120)"
    else
        say "host fault, but could not identify the machine_id to skip"
    fi
done

say "gave up after $ATTEMPTS hosts, all of which failed to start"
exit 75
