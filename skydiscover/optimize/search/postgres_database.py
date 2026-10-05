"""PostgreSQL backend for native Optimize.

A connection holds a session advisory lock for one run. Each operation is a
short transaction: reconstruct algorithm state, query candidates, apply changes,
and commit both. No transaction spans external model or evaluation calls.
"""

import copy
import json
import random
import uuid
from collections.abc import MutableMapping
from contextlib import contextmanager
from dataclasses import asdict

from skydiscover.optimize.search.base_database import Program, ProgramDatabase
from skydiscover.optimize.search.persistence.operations import database_operation
from skydiscover.optimize.search.persistence.schema import SCHEMA_VERSION, connect
from skydiscover.optimize.search.persistence.state import decode, encode, json_data, restore_data
from skydiscover.optimize.utils.metrics import get_score


class _Programs(MutableMapping):
    """SQL-backed active population view; never a candidate cache."""

    def __init__(self, db):
        self.db = db

    def __getitem__(self, key):
        with self.db.operation():
            row = self.db._conn.execute(
                "SELECT active FROM skydiscover.memberships WHERE run_id=%s AND revision=%s AND program_id=%s",
                (self.db.run_id, self.db._revision, key),
            ).fetchone()
            if not row or not row[0]:
                raise KeyError(key)
            return self.db._get_any(key)

    def __setitem__(self, key, value):
        if key != value.id:
            raise ValueError("Candidate key must equal its ID")
        with self.db.operation():
            self.db._put(value)
            self.db._conn.execute(
                "INSERT INTO skydiscover.memberships VALUES (%s,%s,%s,true) "
                "ON CONFLICT (run_id,revision,program_id) DO UPDATE SET active=true",
                (self.db.run_id, self.db._revision, key),
            )

    def __delitem__(self, key):
        with self.db.operation():
            result = self.db._conn.execute(
                "UPDATE skydiscover.memberships SET active=false WHERE run_id=%s AND revision=%s AND program_id=%s AND active",
                (self.db.run_id, self.db._revision, key),
            )
            if not result.rowcount:
                raise KeyError(key)

    def __iter__(self):
        with self.db.operation():
            rows = self.db._conn.execute(
                "SELECT p.id FROM skydiscover.programs p JOIN skydiscover.memberships m "
                "ON (p.run_id=m.run_id AND p.id=m.program_id) "
                "WHERE p.run_id=%s AND m.revision=%s AND m.active ORDER BY p.ordinal",
                (self.db.run_id, self.db._revision),
            ).fetchall()
            return iter([r[0] for r in rows])

    def __len__(self):
        return self.db.count()


class PostgresProgramDatabase(ProgramDatabase):
    durable = True
    # These are infrastructure, not algorithm state.
    _TRANSIENT = {
        "name",
        "config",
        "run_id",
        "programs",
        "prompts_by_program",
        "checkpoint_manager",
        "_program_class",
        "_pareto_front_cache",
        "_pareto_front_cache_valid",
        "_global_pareto_cache",
        "_global_pareto_cache_valid",
    }

    def __init__(self, name, config, **kwargs):
        super().__init__(name, config)
        self._ready = False
        self._depth = 0
        self._identity = {}
        self._originals = {}
        self._state_keys = set()
        self._conn = getattr(config, "_postgres_connection", None)
        self._owns_connection = self._conn is None
        self.run_id = str(uuid.UUID(config.run_id)) if config.run_id else str(uuid.uuid4())
        self._revision = getattr(config, "_strategy_revision", 0)
        self.rng = random.Random(config.random_seed)
        self.programs = _Programs(self)
        self.language = "python"
        self.last_iteration = 0
        self.initial_program_id = None
        self.initial_program_score = None
        self.best_program_id = None
        self._pareto_front_cache = None
        self._pareto_front_cache_valid = False
        self.prompts_by_program = None

    def _open(self):
        if not isinstance(self.programs, _Programs):
            raise TypeError(
                "PostgreSQL strategies must retain the inherited SQL population view; do not replace self.programs"
            )
        if self._conn is None:
            if not self.config.postgres_dsn:
                raise ValueError(
                    "PostgreSQL requires search.database.postgres_dsn or SKYDISCOVER_POSTGRES_DSN"
                )
            self._conn = connect(self.config.postgres_dsn)
        try:
            exists = self._conn.execute(
                "SELECT to_regclass('skydiscover.schema_version')"
            ).fetchone()[0]
            if not exists:
                raise ValueError("PostgreSQL schema is missing; run skydiscover db migrate first")
            version = self._conn.execute(
                "SELECT version FROM skydiscover.schema_version"
            ).fetchone()
            if not version or version[0] != SCHEMA_VERSION:
                raise ValueError(
                    "Unsupported PostgreSQL schema version; migrate the database first"
                )
            if self._owns_connection:
                locked = self._conn.execute(
                    "SELECT pg_try_advisory_lock(hashtextextended(%s,0))", (self.run_id,)
                ).fetchone()[0]
                if not locked:
                    raise RuntimeError(f"Run {self.run_id} is already open in another process")
            with self._conn.transaction():
                row = self._conn.execute(
                    "SELECT search_type,config,active_revision FROM skydiscover.runs WHERE id=%s",
                    (self.run_id,),
                ).fetchone()
                cfg = asdict(self.config)
                # Connection secrets and resume IDs are not durable configuration.
                cfg.pop("postgres_dsn", None)
                cfg.pop("run_id", None)
                if row is None:
                    if self.config.run_id and self._owns_connection:
                        raise ValueError(f"Unknown run ID {self.run_id}")
                    self._conn.execute(
                        "INSERT INTO skydiscover.runs(id,search_type,config) VALUES (%s,%s,%s::jsonb)",
                        (self.run_id, self.name, json.dumps(cfg)),
                    )
                    self._conn.execute(
                        "INSERT INTO skydiscover.strategies(run_id,revision) VALUES (%s,0)",
                        (self.run_id,),
                    )
                elif self._owns_connection:
                    if row[0] != self.name:
                        raise ValueError(f"Run uses {row[0]}, not {self.name}")
                    if row[1] != cfg:
                        raise ValueError(
                            "Search database configuration differs from the stored run"
                        )
                    self._revision = row[2]
                self._ready = True
                with self.operation():
                    pass
        except BaseException:
            self._ready = False
            if self._owns_connection:
                self._conn.close()
            raise

    def _algorithm_state(self):
        return {
            k: v
            for k, v in vars(self).items()
            if k not in self._TRANSIENT
            and not callable(v)
            and k
            not in {
                "_ready",
                "_depth",
                "_identity",
                "_originals",
                "_state_keys",
                "_conn",
                "_owns_connection",
                "_revision",
            }
        }

    @contextmanager
    def operation(self):
        if not self._ready or self._depth:
            yield
            return
        self._depth = 1
        self._identity = {}
        self._originals = {}
        try:
            with self._conn.transaction():
                row = self._conn.execute(
                    "SELECT state FROM skydiscover.strategies WHERE run_id=%s AND revision=%s FOR UPDATE",
                    (self.run_id, self._revision),
                ).fetchone()
                if row is None:
                    raise RuntimeError("Missing search strategy revision")
                if row[0]:
                    state = decode(row[0], self)
                    for key in self._state_keys - state.keys():
                        self.__dict__.pop(key, None)
                    self.__dict__.update(state)
                    self._state_keys = set(state)
                self._pareto_front_cache_valid = False
                self._global_pareto_cache_valid = False
                yield
                snapshot = encode(self._algorithm_state(), self)
                # Persist edits to objects fetched during this operation. Identity
                # lasts only for the transaction and is discarded at its end.
                for pid, program in list(self._identity.items()):
                    if program.to_dict() != self._originals.get(pid):
                        self._put(program)
                self._conn.execute(
                    "UPDATE skydiscover.strategies SET state=%s::jsonb WHERE run_id=%s AND revision=%s",
                    (json.dumps(snapshot, allow_nan=False), self.run_id, self._revision),
                )
        finally:
            self._identity = {}
            self._originals = {}
            self._depth = 0

    def _put(self, program):
        data = json.dumps(json_data(program.to_dict()), allow_nan=False)
        score = get_score(program.metrics)
        combined = program.metrics.get("combined_score")
        combined = float(combined) if isinstance(combined, (int, float)) else None
        self._conn.execute(
            "INSERT INTO skydiscover.programs(run_id,id,parent_id,iteration,score,combined_score,data) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s::jsonb) ON CONFLICT (run_id,id) DO UPDATE SET "
            "parent_id=excluded.parent_id,iteration=excluded.iteration,score=excluded.score,combined_score=excluded.combined_score,data=skydiscover.programs.data || excluded.data",
            (
                self.run_id,
                program.id,
                program.parent_id,
                program.iteration_found,
                score,
                combined,
                data,
            ),
        )
        self._identity[program.id] = program
        self._originals[program.id] = copy.deepcopy(program.to_dict())

    def _ensure_program(self, program):
        row = self._conn.execute(
            "SELECT 1 FROM skydiscover.programs WHERE run_id=%s AND id=%s",
            (self.run_id, program.id),
        ).fetchone()
        if not row:
            self._put(program)
        else:
            self._identity.setdefault(program.id, program)

    def _get_any(self, pid):
        if pid in self._identity:
            return self._identity[pid]
        row = self._conn.execute(
            "SELECT data FROM skydiscover.programs WHERE run_id=%s AND id=%s", (self.run_id, pid)
        ).fetchone()
        if not row:
            return None
        cls = getattr(self, "_program_class", Program)
        program = cls.from_dict(restore_data(row[0]))
        self._identity[pid] = program
        self._originals[pid] = copy.deepcopy(program.to_dict())
        return program

    @database_operation
    def get(self, program_id):
        # Full history remains available even after population eviction.
        return self._get_any(program_id)

    @database_operation
    def count(self):
        return self._conn.execute(
            "SELECT count(*) FROM skydiscover.memberships WHERE run_id=%s AND revision=%s AND active",
            (self.run_id, self._revision),
        ).fetchone()[0]

    @database_operation
    def iter_programs(self):
        return iter([self._get_any(pid) for pid in self.programs])

    @database_operation
    def update(self, program):
        if self._get_any(program.id) is None:
            raise KeyError(program.id)
        self._put(program)
        self._invalidate_pareto_cache()

    @database_operation
    def get_top_programs(self, n=10, metric=None):
        if n < 0:
            raise ValueError("n must be nonnegative")
        if metric or self.is_multiobjective_enabled():
            return super().get_top_programs(n, metric)
        rows = self._conn.execute(
            "SELECT p.id FROM skydiscover.programs p JOIN skydiscover.memberships m ON (p.run_id=m.run_id AND p.id=m.program_id) "
            "WHERE p.run_id=%s AND m.revision=%s AND m.active ORDER BY p.score DESC,p.ordinal LIMIT %s",
            (self.run_id, self._revision, n),
        ).fetchall()
        return [self._get_any(r[0]) for r in rows]

    @database_operation
    def log_prompt(self, program_id, template_key, prompt, responses=None):
        if not self.config.log_prompts:
            return
        data = dict(prompt, responses=responses or [])
        self._conn.execute(
            "INSERT INTO skydiscover.prompts VALUES (%s,%s,%s,%s::jsonb) "
            "ON CONFLICT (run_id,program_id,template_key) DO UPDATE SET data=excluded.data",
            (self.run_id, program_id, template_key, json.dumps(data)),
        )

    @database_operation
    def get_prompts(self, program_id):
        return dict(
            self._conn.execute(
                "SELECT template_key,data FROM skydiscover.prompts WHERE run_id=%s AND program_id=%s",
                (self.run_id, program_id),
            ).fetchall()
        )

    @database_operation
    def sample_for_iteration(self, iteration, num_context_programs=4, **kwargs):
        row = self._conn.execute(
            "SELECT selection FROM skydiscover.attempts WHERE run_id=%s AND iteration=%s",
            (self.run_id, iteration),
        ).fetchone()
        if row and row[0] is not None:
            return decode(row[0], self)
        selection = self.sample(num_context_programs, **kwargs)
        parent = selection[0]
        if isinstance(parent, dict):
            if len(parent) != 1:
                raise ValueError(f"sample() must return exactly one parent, got {len(parent)}")
            parent = next(iter(parent.values()))
        encoded_selection = json.dumps(encode(selection, self))
        self._conn.execute(
            "INSERT INTO skydiscover.attempts(run_id,iteration,parent_id,selection) VALUES (%s,%s,%s,%s::jsonb) "
            "ON CONFLICT (run_id,iteration) DO UPDATE SET parent_id=excluded.parent_id,selection=excluded.selection",
            (self.run_id, iteration, parent.id if parent is not None else None, encoded_selection),
        )
        return selection

    @database_operation
    def is_iteration_complete(self, iteration):
        row = self._conn.execute(
            "SELECT completed FROM skydiscover.attempts WHERE run_id=%s AND iteration=%s",
            (self.run_id, iteration),
        ).fetchone()
        return bool(row and row[0])

    @database_operation
    def complete_iteration(self, iteration, controller_state=None, error=None, outcome=None):
        self._conn.execute(
            "INSERT INTO skydiscover.attempts(run_id,iteration,completed,error,outcome) VALUES (%s,%s,true,%s,%s::jsonb) "
            "ON CONFLICT (run_id,iteration) DO UPDATE SET completed=true,error=excluded.error,outcome=coalesce(excluded.outcome,skydiscover.attempts.outcome)",
            (
                self.run_id,
                iteration,
                error,
                json.dumps(json_data(outcome)) if outcome is not None else None,
            ),
        )
        self.last_iteration = max(self.last_iteration, iteration)
        next_iteration = self.next_iteration
        while self.is_iteration_complete(next_iteration):
            next_iteration += 1
        self._conn.execute(
            "UPDATE skydiscover.runs SET next_iteration=%s WHERE id=%s",
            (next_iteration, self.run_id),
        )
        if controller_state is not None:
            self._conn.execute(
                "UPDATE skydiscover.runs SET controller_state=%s::jsonb WHERE id=%s",
                (json.dumps(encode(controller_state, self)), self.run_id),
            )

    @property
    def next_iteration(self):
        return self._conn.execute(
            "SELECT next_iteration FROM skydiscover.runs WHERE id=%s", (self.run_id,)
        ).fetchone()[0]

    @database_operation
    def get_controller_state(self):
        state = self._conn.execute(
            "SELECT controller_state FROM skydiscover.runs WHERE id=%s", (self.run_id,)
        ).fetchone()[0]
        return decode(state, self) if state else {}

    @database_operation
    def save_controller_state(self, state):
        self._conn.execute(
            "UPDATE skydiscover.runs SET controller_state=%s::jsonb WHERE id=%s",
            (json.dumps(encode(state, self)), self.run_id),
        )

    @database_operation
    def set_strategy_source(self, source):
        self._conn.execute(
            "UPDATE skydiscover.strategies SET source=%s WHERE run_id=%s AND revision=%s",
            (source, self.run_id, self._revision),
        )

    @database_operation
    def activate_strategy(self):
        self._conn.execute(
            "UPDATE skydiscover.runs SET active_revision=%s WHERE id=%s",
            (self._revision, self.run_id),
        )

    def strategy_config(self, revision=None):
        config = copy.copy(self.config)
        config.run_id = self.run_id
        config._postgres_connection = self._conn
        config._strategy_revision = (
            revision
            if revision is not None
            else self._conn.execute(
                "SELECT coalesce(max(revision),-1)+1 FROM skydiscover.strategies WHERE run_id=%s",
                (self.run_id,),
            ).fetchone()[0]
        )
        if revision is None:
            self._conn.execute(
                "INSERT INTO skydiscover.strategies(run_id,revision) VALUES (%s,%s)",
                (self.run_id, config._strategy_revision),
            )
        return config

    def _save_program(self, *args, **kwargs):
        # Algorithms retain this compatibility hook; PostgreSQL already persisted it.
        pass

    def save(self, *args, **kwargs):
        raise ValueError("PostgreSQL runs use run IDs, not file checkpoints")

    def load(self, *args, **kwargs):
        raise ValueError("Resume PostgreSQL runs using their run ID")

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._ready = False
