"""The only PostgreSQL schema definition. Algorithm subclasses do not own DDL."""

SCHEMA_VERSION = 1
DDL = """
CREATE SCHEMA IF NOT EXISTS skydiscover;
CREATE TABLE IF NOT EXISTS skydiscover.schema_version (
    singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton), version integer NOT NULL
);
CREATE TABLE IF NOT EXISTS skydiscover.runs (
    id uuid PRIMARY KEY, search_type text NOT NULL, config jsonb NOT NULL,
    next_iteration integer NOT NULL DEFAULT 0,
    active_revision integer NOT NULL DEFAULT 0,
    controller_state jsonb NOT NULL DEFAULT '{}',
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS skydiscover.strategies (
    run_id uuid NOT NULL REFERENCES skydiscover.runs(id) ON DELETE CASCADE,
    revision integer NOT NULL, source text, state jsonb NOT NULL DEFAULT '{}',
    PRIMARY KEY (run_id, revision)
);
CREATE TABLE IF NOT EXISTS skydiscover.programs (
    run_id uuid NOT NULL REFERENCES skydiscover.runs(id) ON DELETE CASCADE,
    id text NOT NULL, parent_id text, iteration integer NOT NULL,
    score double precision NOT NULL, combined_score double precision,
    data jsonb NOT NULL, ordinal bigint GENERATED ALWAYS AS IDENTITY,
    PRIMARY KEY (run_id, id)
);
CREATE INDEX IF NOT EXISTS programs_rank ON skydiscover.programs(run_id, score DESC, ordinal);
CREATE TABLE IF NOT EXISTS skydiscover.memberships (
    run_id uuid NOT NULL, revision integer NOT NULL, program_id text NOT NULL,
    active boolean NOT NULL DEFAULT true,
    PRIMARY KEY (run_id, revision, program_id),
    FOREIGN KEY (run_id, revision) REFERENCES skydiscover.strategies(run_id, revision) ON DELETE CASCADE,
    FOREIGN KEY (run_id, program_id) REFERENCES skydiscover.programs(run_id, id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS skydiscover.prompts (
    run_id uuid NOT NULL REFERENCES skydiscover.runs(id) ON DELETE CASCADE,
    program_id text NOT NULL, template_key text NOT NULL, data jsonb NOT NULL,
    PRIMARY KEY (run_id, program_id, template_key)
);
CREATE TABLE IF NOT EXISTS skydiscover.attempts (
    run_id uuid NOT NULL REFERENCES skydiscover.runs(id) ON DELETE CASCADE,
    iteration integer NOT NULL, parent_id text, selection jsonb,
    completed boolean NOT NULL DEFAULT false,
    error text, outcome jsonb, PRIMARY KEY (run_id, iteration),
    FOREIGN KEY (run_id, parent_id) REFERENCES skydiscover.programs(run_id, id)
);
"""


def connect(dsn):
    try:
        import psycopg
    except ImportError as exc:
        raise ImportError(
            "PostgreSQL requires the optional driver: uv sync --extra postgres"
        ) from exc
    return psycopg.connect(dsn, autocommit=True)


def migrate(dsn):
    with connect(dsn) as conn, conn.transaction():
        # Serialize concurrent migrations, independently of run locks.
        conn.execute("SELECT pg_advisory_xact_lock(781004100)")
        conn.execute("CREATE SCHEMA IF NOT EXISTS skydiscover")
        exists = conn.execute("SELECT to_regclass('skydiscover.schema_version')").fetchone()[0]
        if exists:
            row = conn.execute("SELECT version FROM skydiscover.schema_version").fetchone()
            if row and row[0] != SCHEMA_VERSION:
                raise ValueError(f"Unsupported database schema version {row[0]}")
        conn.execute(DDL)
        conn.execute(
            "INSERT INTO skydiscover.schema_version VALUES (true, %s) ON CONFLICT DO NOTHING",
            (SCHEMA_VERSION,),
        )


def main(argv=None, prog=None):
    import argparse
    import os

    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument("action", choices=["migrate"])
    parser.add_argument("--dsn", default=os.environ.get("SKYDISCOVER_POSTGRES_DSN"))
    args = parser.parse_args(argv)
    if not args.dsn:
        parser.error("set SKYDISCOVER_POSTGRES_DSN or pass --dsn")
    migrate(args.dsn)
    print(f"PostgreSQL schema is at version {SCHEMA_VERSION}")
    return 0
