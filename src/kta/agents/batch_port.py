"""The batch-environment port the experiment process uses (WP11, D403).

The orchestrator (layer 7) may not import agents (layer 8), so it resolves the interface
``batch_envs`` from the registry; this is the implementation: ``kta agents env build``
when the batch does not exist yet, then ``kta agents env verify`` (D322). A batch whose
lock names another profile, or that fails verification, is refused.
"""

from __future__ import annotations

from typing import Any

from kta.agents import batches
from kta.core.context import Context
from kta.core.errors import PermissionDenied
from kta.core.registry import resolve
from kta.store import code_state


class BatchEnvs:
    def __init__(self, ctx: Context) -> None:
        self.store = resolve("store", ctx)
        self.data_dir = ctx.data_dir

    def ensure(self, batch: str, profile_id: str) -> dict[str, Any]:
        """Build ``batch`` from ``profile_id`` if absent, verify it, return its lock."""
        root = batches.batches_root(self.data_dir) / batch
        built = False
        if not root.exists():
            batches.build(self.store, self.data_dir, batch, profile_id, code_state().commit)
            built = True
        lock = batches.require_verified(self.store, root)
        if lock["profile"] != profile_id:
            raise PermissionDenied(
                f"batch {batch} was built from {lock['profile']}, not {profile_id}"
            )
        return {
            "batch": batch,
            "root": str(root),
            "profile": profile_id,
            "commit": lock["commit"],
            "built": built,
        }
