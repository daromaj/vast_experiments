#!/usr/bin/env bash
# Dead-man's switch: destroy a vast.ai instance at a hard deadline unless it has
# already been destroyed cleanly.
#
# An unattended run that stalls - a hung ssh, a wedged agent, a closed laptop -
# leaves the rental billing indefinitely. This detaches from the session with
# setsid so it outlives whatever started it.
#
#   ./vast_deadman.sh 45944385 10800     # destroy in at most 3 hours
#
# Safe to leave running after a clean destroy: it re-checks liveness and exits
# without acting if the instance is already gone. Targets one instance id, and
# vast.ai does not recycle ids, so it cannot hit a later rental.
set -uo pipefail

INSTANCE="${1:?usage: vast_deadman.sh <instance_id> <seconds>}"
DEADLINE_S="${2:-10800}"
LOG="${DEADMAN_LOG:-/tmp/vast_deadman_${INSTANCE}.log}"

exec >>"$LOG" 2>&1
echo "=== deadman armed for $INSTANCE, ${DEADLINE_S}s, at $(date -u +%FT%TZ) ==="

end=$(( $(date +%s) + DEADLINE_S ))
while [[ $(date +%s) -lt $end ]]; do
    if ! vastai show instance "$INSTANCE" --raw >/dev/null 2>&1; then
        echo "$(date -u +%FT%TZ) instance gone - standing down"
        exit 0
    fi
    sleep 120
done

echo "$(date -u +%FT%TZ) DEADLINE REACHED - destroying $INSTANCE"
yes y | vastai destroy instance "$INSTANCE"
sleep 10
vastai show instances --raw 2>/dev/null | grep -c "$INSTANCE" || true
echo "$(date -u +%FT%TZ) deadman done"
