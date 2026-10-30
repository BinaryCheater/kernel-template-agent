-- KTA store schema v1 (spec/modules/core-store.md "表").
-- Evidence tables are append-only; operations tables may change and log to ops_events.

CREATE TABLE records (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    schema INT NOT NULL,
    created_at TEXT NOT NULL,
    actor_type TEXT NOT NULL,
    actor_name TEXT NOT NULL,
    task TEXT,
    origin TEXT NOT NULL,
    json TEXT NOT NULL
);
CREATE INDEX records_kind_created ON records(kind, created_at);
CREATE INDEX records_kind_name ON records(kind, json_extract(json, '$.body.name'), created_at);
CREATE INDEX records_task ON records(task);

CREATE TRIGGER records_no_update BEFORE UPDATE ON records
BEGIN
    SELECT RAISE(ABORT, 'records are append-only: UPDATE is not allowed');
END;
CREATE TRIGGER records_no_delete BEFORE DELETE ON records
BEGIN
    SELECT RAISE(ABORT, 'records are append-only: DELETE is not allowed');
END;

CREATE TABLE edges (
    src TEXT NOT NULL,
    rel TEXT NOT NULL,
    dst TEXT NOT NULL,
    PRIMARY KEY (src, rel, dst)
) WITHOUT ROWID;
CREATE INDEX edges_dst ON edges(dst, rel);

CREATE TRIGGER edges_no_update BEFORE UPDATE ON edges
BEGIN
    SELECT RAISE(ABORT, 'edges are append-only: UPDATE is not allowed');
END;
CREATE TRIGGER edges_no_delete BEFORE DELETE ON edges
BEGIN
    SELECT RAISE(ABORT, 'edges are append-only: DELETE is not allowed');
END;

CREATE TABLE blobs (
    sha256 TEXT PRIMARY KEY,
    size INT NOT NULL,
    media TEXT NOT NULL,
    first_seen TEXT NOT NULL
);

CREATE VIRTUAL TABLE records_fts USING fts5(
    id UNINDEXED,
    kind UNINDEXED,
    title,
    text,
    tokenize = 'unicode61'
);
-- Default ranking: bm25 with column weights id, kind (unindexed), title, text. Using the
-- built-in rank lets FTS5 sort internally, so snippets are built only for returned rows.
INSERT INTO records_fts(records_fts, rank) VALUES ('rank', 'bm25(0.0, 0.0, 4.0, 1.0)');

-- Operations tables (mutable, not evidence).
CREATE TABLE tasks (
    id TEXT PRIMARY KEY,
    card_id TEXT NOT NULL,
    card_type TEXT NOT NULL,
    state TEXT NOT NULL,
    priority INT NOT NULL,
    lease_owner TEXT,
    lease_until TEXT,
    attempts INT NOT NULL DEFAULT 0,
    work_unit TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX tasks_ready ON tasks(state, card_type, priority DESC, created_at);

CREATE TABLE loops (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    subject TEXT NOT NULL,
    state TEXT NOT NULL,
    data_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE workers (
    name TEXT PRIMARY KEY,
    host TEXT NOT NULL,
    healthy INT NOT NULL,
    heartbeat TEXT NOT NULL
);

CREATE TABLE locks (
    name TEXT PRIMARY KEY,
    holder TEXT NOT NULL,
    until TEXT NOT NULL
);

CREATE TABLE pointers (
    name TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE ops_events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    at TEXT NOT NULL,
    table_name TEXT NOT NULL,
    row_id TEXT NOT NULL,
    change_json TEXT NOT NULL
);

CREATE TABLE spans (
    span_id TEXT PRIMARY KEY,
    parent_id TEXT,
    trace_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    lane TEXT NOT NULL,
    name TEXT NOT NULL,
    start_ts TEXT NOT NULL,
    end_ts TEXT,
    status TEXT NOT NULL,
    attrs_json TEXT NOT NULL
);
CREATE INDEX spans_start ON spans(start_ts);
CREATE INDEX spans_trace ON spans(trace_id);
CREATE INDEX spans_lane ON spans(lane, start_ts);

CREATE TABLE budgets (
    scope TEXT PRIMARY KEY,
    limit_json TEXT NOT NULL,
    used_json TEXT NOT NULL,
    reserved_json TEXT NOT NULL
);
