#!/bin/bash
#
# Thin wrapper over the vastai CLI for the provisioning-optimisation loop:
# find the running instance, push a script to it, run it, pull results back.
#
# Every subcommand takes an optional instance id as its LAST argument. Omit it and
# the single running instance is used; if there is more than one, you are asked to
# name it rather than having one picked for you.
#
#   scripts/vast.sh ls                      # instances + status
#   scripts/vast.sh ssh [id]                # interactive shell
#   scripts/vast.sh run "nvidia-smi" [id]   # run one command
#   scripts/vast.sh push local.py [id]      # copy a file to /workspace/
#   scripts/vast.sh pull /workspace/x [id]  # copy a file back to ./
#   scripts/vast.sh log [id]                # fetch provisioning.log + analyse it
#   scripts/vast.sh probe [id]              # push + run the SageAttention probe
#   scripts/vast.sh watch [id]              # follow provisioning as it happens
#
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

die() { echo "error: $*" >&2; exit 1; }

command -v vastai >/dev/null || die "vastai CLI not found in PATH"

resolve_instance() {
    # Echo the instance id to act on. An explicit id always wins.
    if [[ -n ${1:-} ]]; then echo "$1"; return; fi

    local ids
    ids=$(vastai show instances --raw 2>/dev/null |
        python3 -c "
import json,sys
try: d=json.load(sys.stdin)
except Exception: sys.exit(0)
print('\n'.join(str(i['id']) for i in d))
")
    local count
    count=$(echo "$ids" | grep -c . || true)
    [[ $count -eq 0 ]] && die "no instances running"
    # Refuse to guess when it is ambiguous — the wrong box wastes a rental.
    [[ $count -gt 1 ]] && die "$count instances running, pass an id explicitly:"$'\n'"$ids"
    echo "$ids"
}

ssh_target() {
    # vastai ssh-url gives ssh://root@host:port — split it into "host port".
    local id="$1" url
    url=$(vastai ssh-url "$id" 2>/dev/null) || die "could not get ssh url for $id"
    [[ $url =~ ssh://([^@]+)@([^:]+):([0-9]+) ]] ||
        die "unexpected ssh url format: $url"
    echo "${BASH_REMATCH[1]} ${BASH_REMATCH[2]} ${BASH_REMATCH[3]}"
}

# StrictHostKeyChecking=no: every rental is a brand new host key, and the prompt
# would block non-interactive use. UserKnownHostsFile=/dev/null keeps the churn
# out of the real known_hosts.
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR)

do_ssh() {
    local id="$1"; shift
    read -r user host port <<<"$(ssh_target "$id")"
    if [[ $# -eq 0 ]]; then
        ssh "${SSH_OPTS[@]}" -p "$port" "${user}@${host}"
    else
        ssh "${SSH_OPTS[@]}" -p "$port" "${user}@${host}" "$@"
    fi
}

do_push() {
    local id="$1" src="$2" dest="${3:-/workspace/}"
    [[ -f $src ]] || die "no such file: $src"
    read -r user host port <<<"$(ssh_target "$id")"
    scp "${SSH_OPTS[@]}" -P "$port" "$src" "${user}@${host}:${dest}" ||
        die "scp failed"
    echo "pushed $(basename "$src") -> ${dest}"
}

do_pull() {
    local id="$1" src="$2" dest="${3:-.}"
    read -r user host port <<<"$(ssh_target "$id")"
    scp "${SSH_OPTS[@]}" -P "$port" "${user}@${host}:${src}" "$dest" ||
        die "scp failed (does $src exist on the instance?)"
    echo "pulled $src -> $dest"
}

cmd="${1:-help}"; shift || true

case "$cmd" in
    ls)
        vastai show instances --raw 2>/dev/null | python3 -c '
import json,sys
try: d=json.load(sys.stdin)
except Exception: d=[]
if not d: print("no instances"); raise SystemExit
for i in d:
    # `or` not a .get default: a freshly created instance has these keys
    # present but null, so the default never fires.
    row = "{}  {:<10} {:<16} ${:.3f}/hr  {}".format(
        i["id"], i.get("actual_status") or "?", i.get("gpu_name") or "?",
        i.get("dph_total") or 0, i.get("status_msg") or "")
    print(row[:110])
'
        ;;

    ssh)
        id=$(resolve_instance "${1:-}") || exit 1
        do_ssh "$id"
        ;;

    run)
        [[ $# -ge 1 ]] || die "usage: vast.sh run \"<command>\" [id]"
        remote_cmd="$1"; shift
        id=$(resolve_instance "${1:-}") || exit 1
        do_ssh "$id" "$remote_cmd"
        ;;

    push)
        [[ $# -ge 1 ]] || die "usage: vast.sh push <local-file> [id]"
        src="$1"; shift
        id=$(resolve_instance "${1:-}") || exit 1
        do_push "$id" "$src"
        ;;

    pull)
        [[ $# -ge 1 ]] || die "usage: vast.sh pull <remote-path> [id]"
        src="$1"; shift
        id=$(resolve_instance "${1:-}") || exit 1
        do_pull "$id" "$src"
        ;;

    log)
        id=$(resolve_instance "${1:-}") || exit 1
        out="provisioning-${id}.log"
        do_pull "$id" "/workspace/provisioning.log" "$out"
        echo
        python3 "${REPO}/scripts/parse_provision_log.py" "$out"
        ;;

    probe)
        id=$(resolve_instance "${1:-}") || exit 1
        do_push "$id" "${REPO}/scripts/sage_abi_probe.py"
        # The venv matters: sage lives in /venv/main, not the system python.
        do_ssh "$id" "source /venv/main/bin/activate && python3 /workspace/sage_abi_probe.py"
        ;;

    watch)
        id=$(resolve_instance "${1:-}") || exit 1
        do_ssh "$id" "tail -f -n +1 /workspace/provisioning.log" |
            grep --line-buffered -E "PHASE|DL_TIME|SAGE_|PROGRESS|WARNING|ERROR"
        ;;

    help|--help|-h)
        sed -n '/^# Thin wrapper/,/^#   scripts\/vast.sh watch/p' "${BASH_SOURCE[0]}" |
            sed 's/^# \?//'
        ;;

    *)
        die "unknown command: $cmd (try: vast.sh help)"
        ;;
esac
