"""External native-CLI boundary for watchdog tests; never calls a model."""

import json
import signal
import sys
import time
from pathlib import Path

home = Path(sys.argv[1])
(home / "sessions").mkdir()
(home / "sessions/rollout-native-test.jsonl").write_text(
    json.dumps({"type": "session_meta", "payload": {"id": "native-test"}}) + "\n"
)


def interrupted(signum, frame):
    (home / "interrupted").write_text("persisted")
    raise SystemExit(0)


signal.signal(signal.SIGINT, interrupted)
print(json.dumps({"type": "thread.started", "thread_id": "native-test"}), flush=True)
if len(sys.argv) > 2:
    ledger = Path(sys.argv[2])
    began = time.time()
    phase = "queued"

    def event(name):
        with ledger.open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "event": name,
                        "at_s": time.time(),
                        "request_id": "queue-test",
                        "submission_id": "one",
                        "slot": "run-01",
                        "worker": "boundary-test",
                        "job_id": "job-test",
                    }
                )
                + "\n"
            )

    event("blocked")
    event("queued")

    def tick(signum, frame):
        global phase
        elapsed = time.time() - began
        if elapsed < 0.8:
            event("waiting")
        elif phase == "queued":
            event("acquired")
            event("completed")
            phase = "done"
        elif elapsed > 1.0:
            raise SystemExit(0)

    signal.signal(signal.SIGALRM, tick)
    signal.setitimer(signal.ITIMER_REAL, 0.02, 0.02)
while True:
    signal.pause()
