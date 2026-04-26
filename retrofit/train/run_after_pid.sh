#!/bin/bash
# Wait for <wait-pid> to exit, then exec the rest of the args as a command.
# Usage: bash run_after_pid.sh <wait-pid> <cmd...>
set -u
WAIT_PID="${1:?pid}"
shift
LOG_TAG="after_pid_${WAIT_PID}"
echo "[$LOG_TAG] waiting for pid $WAIT_PID at $(date)"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
echo "[$LOG_TAG] pid $WAIT_PID exited at $(date), exec follow-up"
exec "$@"
