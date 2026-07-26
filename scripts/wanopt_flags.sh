#!/usr/bin/env bash
# Turn the multitalk_loop.py patch flags on or off. Run this ON the instance.
#
# The patch reads os.environ at call time, so the variables have to exist in the
# ComfyUI process itself - exporting them in an ssh shell does nothing. ComfyUI
# is supervisor-managed, so the only place that works is the environment= line
# in its program config, followed by a reread/update/restart.
#
#   wanopt_flags.sh on    both optimizations active
#   wanopt_flags.sh off   stock behaviour (patch stays in place but inert)
#
# Keeping the patch installed and toggling only the environment means an A/B
# compares two runs of the same file, so a difference cannot be a patching
# artefact.
set -u
CONF=/etc/supervisor/conf.d/comfyui.conf
BASE='environment=PROC_NAME="%(program_name)s"'

mode="${1:-}"
case "$mode" in
    on)  line="${BASE},WANOPT_Y_CACHE=\"1\",WANOPT_KEEP_CACHE_WARM=\"1\"" ;;
    off) line="$BASE" ;;
    *)   echo "usage: wanopt_flags.sh on|off"; exit 1 ;;
esac

python3 - "$CONF" "$line" <<'PY'
import re, sys
conf, line = sys.argv[1], sys.argv[2]
src = open(conf).read()
new, n = re.subn(r'^environment=.*$', line.replace('\\', '\\\\'), src, count=1, flags=re.M)
if n != 1:
    print(f"FATAL: expected exactly 1 environment= line, found {n}")
    raise SystemExit(1)
open(conf, "w").write(new)
print("set:", line)
PY
[[ $? -eq 0 ]] || exit 1

supervisorctl reread >/dev/null
supervisorctl update >/dev/null
supervisorctl restart comfyui >/dev/null

# Wait for it to actually answer before returning, otherwise the caller races it.
for _ in $(seq 1 60); do
    curl -sf http://127.0.0.1:18188/system_stats >/dev/null && break
    sleep 5
done

echo "comfyui restarted with WANOPT flags ${mode}"
grep '^environment=' "$CONF"
