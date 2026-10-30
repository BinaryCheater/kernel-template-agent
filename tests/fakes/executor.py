"""An Executor stand-in for the agent runner's cancel watcher (WP6 round 3, D374).

It accepts jobs, keeps them unfinished until ``finish`` or ``cancel``, and records the
cancelled IDs. ``cancel(job_id) -> None`` is the signature the WP2 session is adding to
``JobExecutor``; tests/contract/test_executor_contract.py compares the signatures.
"""

from __future__ import annotations

import threading

from kta.core.jobs import Job, JobResult


class RecordingExecutor:
    FIXTURE = True

    def __init__(self) -> None:
        self.submitted: list[str] = []
        self.cancelled: list[str] = []
        self.any_cancelled = threading.Event()
        self._n = 0
        self._blobs: dict[str, bytes] = {}

    def submit(self, job: Job) -> str:
        self._n += 1
        job_id = f"job{self._n}"
        self.submitted.append(job_id)
        return job_id

    def wait(self, job_id: str) -> JobResult:
        raise NotImplementedError(
            "no job ever finishes here; tests call cancel"
        )  # kta:allowed test double

    def run(self, job: Job) -> JobResult:
        return self.wait(self.submit(job))

    def read_blob(self, sha256: str) -> bytes:
        return self._blobs[sha256]

    def put_blob(self, data: bytes, media: str) -> str:
        key = f"{len(self._blobs):064x}"
        self._blobs[key] = data
        return key

    def cancel(self, job_id: str) -> None:
        self.cancelled.append(job_id)
        self.any_cancelled.set()
