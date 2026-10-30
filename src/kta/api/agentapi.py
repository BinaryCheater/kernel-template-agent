"""Token endpoints for the agent-facing ``kta`` commands in client mode (WP6 D370, D352).

Inside an agent workspace ``kta`` holds a task token, not a store; each command below is the
API side of one command and does what the local command does (``kta.cli.agent_cmds``),
with the same knowledge, orchestrator and checker functions. Every call:

- needs ``Authorization: Bearer <task token>``, verified by WP8's guard for a live task;
- is a span (kind ``cli``, lane ``process:api``, ``attrs.task_id``) under the agent session
  when the client sends ``X-KTA-Trace-Parent`` (its ``KTA_TRACE_PARENT``), so the Task page
  and the Pipeline show the call;
- takes files as ``{"files": {"<relative path>": "<base64>"}}`` (at most 32 MiB in all).

Endpoints: GET /agent/records/{id}, /agent/cards/{id}, /agent/why/{id}; POST /agent/query,
/agent/validate-template, /agent/validate-family, /agent/instantiate, /agent/build,
/agent/sass, /agent/submit.
"""

from __future__ import annotations

import base64
import binascii
import tempfile
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from fastapi import APIRouter, Header, Query
from pydantic import BaseModel, ConfigDict, ValidationError

from kta.api.auth import bearer
from kta.api.common import Api
from kta.checks.sass import sass_check
from kta.core.errors import ConfigError, NotSupported, SchemaError
from kta.core.ids import UlidFactory
from kta.core.schemas import Actor
from kta.core.schemas.evaluation import Case
from kta.core.schemas.knowledge import Template
from kta.core.trace import TRACE_PARENT_VAR, new_tracer, span, tracing
from kta.knowledge import (
    card,
    load_family,
    nearest_chains,
    retrieve_families,
    retrieve_templates,
    validate_family,
    validate_template,
)
from kta.orchestrator.pipeline import (
    Services,
    build,
    family_package,
    generate,
    sass_for,
    templates_of,
)
from kta.orchestrator.tasks import TaskQueue

MAX_UPLOAD_BYTES = 32 * 1024 * 1024
NOTE_FILE = "DELIVERY.md"  # the delivery note inside out/ (as `kta submit`)


class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid")


class QueryBody(_Body):
    what: Literal["pool", "tmpl", "chain"]
    term: str | None = None
    case: str | None = None
    k: int = 5


class TemplateBody(_Body):
    template: dict[str, Any]


class FilesBody(_Body):
    files: dict[str, str]


class GenerateBody(_Body):
    family: str
    assignment: dict[str, Any]
    shape: dict[str, Any]


class SubmitBody(_Body):
    files: dict[str, str]
    note: str


def decode_files(files: Mapping[str, str]) -> dict[str, bytes]:
    """Relative POSIX paths -> bytes; refuses absolute paths, ``..`` and oversize uploads."""
    out: dict[str, bytes] = {}
    total = 0
    for name, data in files.items():
        rel = PurePosixPath(name)
        if rel.is_absolute() or not rel.parts or ".." in rel.parts:
            raise ConfigError(f"file name {name!r} must be a relative path inside the upload")
        try:
            raw = base64.b64decode(data, validate=True)
        except binascii.Error as exc:
            raise ConfigError(f"file {name!r} is not base64: {exc}") from exc
        total += len(raw)
        if total > MAX_UPLOAD_BYTES:
            raise ConfigError(f"upload larger than {MAX_UPLOAD_BYTES} bytes")
        out[rel.as_posix()] = raw
    return out


def _issues(issues: list[Any]) -> dict[str, Any]:
    rows = [
        {"kind": i.kind, "message": i.message, "where": i.where, "severity": i.severity}
        for i in issues
    ]
    return {"issues": rows, "ok": not any(r["severity"] == "error" for r in rows)}


def router(api: Api) -> APIRouter:
    r = APIRouter(prefix="/agent")
    svc = Services(api.ctx, api.store, api.ops)

    @contextmanager
    def call(name: str, authorization: str | None, parent: str | None) -> Generator[str]:
        """Verify the token, then run the call as a span of the agent's trace; yields the task."""
        task = api.orchestration.verify_task_token(bearer(authorization), api.now())
        environ = {TRACE_PARENT_VAR: parent} if parent else {}
        tracer = new_tracer(api.ops, api.store.clock, UlidFactory(api.store.clock), environ)
        with tracing(tracer), span("cli", name, "process:api", task_id=task):
            yield task

    Auth = Header(None)
    Parent = Header(None, alias="X-KTA-Trace-Parent")

    @r.get("/records/{id}")
    def show(
        id: str, authorization: str | None = Auth, parent: str | None = Parent
    ) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        with call(f"kta show {id[:12]}", authorization, parent):
            return {"id": id, **api.store.get_json(id)}

    @r.get("/cards/{id}")
    def get_card(
        id: str, authorization: str | None = Auth, parent: str | None = Parent
    ) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        with call(f"kta card {id[:12]}", authorization, parent):
            c = card(api.store, id)
            return {"id": c.id, "kind": c.kind, "name": c.name, "lines": c.lines}

    @r.get("/why/{id}")
    def why(  # pyright: ignore[reportUnusedFunction]
        id: str,
        depth: int = Query(4, le=20),
        authorization: str | None = Auth,
        parent: str | None = Parent,
    ) -> dict[str, Any]:
        with call(f"kta why {id[:12]}", authorization, parent):
            g = api.store.lineage(id, depth=depth)
            return {"root": g.root, "nodes": g.nodes, "edges": g.edges}

    @r.post("/query")
    def query(
        b: QueryBody, authorization: str | None = Auth, parent: str | None = Parent
    ) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        with call(f"kta query {b.what} {b.term or b.case or ''}"[:80], authorization, parent):
            if b.what == "tmpl":
                cards = retrieve_templates(api.store, text=b.term, k=b.k)
            else:
                if not b.case:
                    raise ConfigError(f"query {b.what} needs the Case ID (case)")
                rec = api.store.get(b.case)
                if rec.kind != "Case":
                    raise NotSupported("query near a family: give the Case ID (case)")
                c = Case.model_validate(rec.body)
                pick = retrieve_families if b.what == "pool" else nearest_chains
                cards = pick(api.store, c, k=b.k)
            return {
                "cards": [
                    {"id": x.id, "kind": x.kind, "name": x.name, "lines": x.lines} for x in cards
                ]
            }

    @r.post("/validate-template")
    def check_template(
        b: TemplateBody, authorization: str | None = Auth, parent: str | None = Parent
    ) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        with call("kta validate-template", authorization, parent):
            body = b.template.get("template", b.template)
            try:
                tpl = Template.model_validate(body)
            except ValidationError as exc:
                raise SchemaError(f"not a Template: {exc}") from exc
            return _issues(validate_template(tpl, api.store))

    @r.post("/validate-family")
    def check_family(
        b: FilesBody, authorization: str | None = Auth, parent: str | None = Parent
    ) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        with call("kta validate-family", authorization, parent):
            files = decode_files(b.files)
            with tempfile.TemporaryDirectory(prefix="kta-api-family-") as tmp:
                for rel, data in files.items():
                    path = Path(tmp) / rel
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
                return _issues(validate_family(load_family(Path(tmp)), api.store))

    @r.post("/instantiate")
    def instantiate(
        b: GenerateBody, authorization: str | None = Auth, parent: str | None = Parent
    ) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        with call(f"kta instantiate {b.family[:12]}", authorization, parent):
            gk = generate(family_package(svc, b.family), b.assignment, b.shape)
            return {
                "files": gk.files,
                "launch": gk.launch.to_dict(),
                "entry": gk.entry,
                "markers": gk.markers,
            }

    @r.post("/build")
    def build_family(
        b: GenerateBody, authorization: str | None = Auth, parent: str | None = Parent
    ) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        with call(f"kta build {b.family[:12]}", authorization, parent):
            gk = generate(family_package(svc, b.family), b.assignment, b.shape)
            result, module = build(svc, gk)
            return {"build": module.build_id, **result.model_dump()}

    @r.post("/sass")
    def sass(
        b: GenerateBody, authorization: str | None = Auth, parent: str | None = Parent
    ) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        with call(f"kta sass {b.family[:12]}", authorization, parent):
            fp = family_package(svc, b.family)
            result, module = build(svc, generate(fp, b.assignment, b.shape))
            if not result.ok or not module.sass_blob:
                raise ConfigError(f"build {module.build_id[:12]} failed or has no SASS")
            expect = sass_for(fp, templates_of(api.store, fp), b.assignment)
            checked = sass_check(svc.checker(), module, expect, b.assignment)
            return {
                "check": checked.id,
                "passed": checked.passed,
                "failures": checked.body.failures,
            }

    @r.post("/submit")
    def submit(
        b: SubmitBody, authorization: str | None = Auth, parent: str | None = Parent
    ) -> dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
        with call("kta submit", authorization, parent) as task:
            if not b.note.strip():
                raise ConfigError("the delivery note is empty")
            files = {k: v for k, v in decode_files(b.files).items() if k != NOTE_FILE}
            q = TaskQueue(api.ctx, api.store, api.ops, svc)
            owner = q.get(task).lease_owner or ""
            sub = q.submit(task, owner, files, b.note, Actor(type="agent", name="kta submit"))
            return {"task": task, "submission": sub, "files": sorted(files)}

    return r


def encode_files(files: Mapping[str, bytes]) -> dict[str, str]:
    """The client side of ``decode_files`` (also used by tests)."""
    return {k: base64.b64encode(v).decode("ascii") for k, v in files.items()}
