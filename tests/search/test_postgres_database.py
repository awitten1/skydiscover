"""Contract and recovery tests against an actual PostgreSQL instance.

Set SKYDISCOVER_TEST_POSTGRES_DSN to an isolated database. Schema migration is
explicit in production; this fixture applies it to the test database.
"""

import copy
import os
import uuid

import pytest

from skydiscover.optimize.config import (
    AdaEvolveDatabaseConfig,
    BeamSearchDatabaseConfig,
    BestOfNDatabaseConfig,
    DatabaseConfig,
    GEPANativeDatabaseConfig,
    OpenEvolveNativeDatabaseConfig,
)
from skydiscover.optimize.search.base_database import Program
from skydiscover.optimize.search.persistence.schema import connect, migrate
from skydiscover.optimize.search.postgres_algorithms import (
    AdaEvolvePostgresProgramDatabase,
    BeamSearchPostgresProgramDatabase,
    BestOfNPostgresProgramDatabase,
    ClaudeCodePostgresProgramDatabase,
    GEPANativePostgresProgramDatabase,
    OpenEvolveNativePostgresProgramDatabase,
    SearchStrategyPostgresProgramDatabase,
    TopKPostgresProgramDatabase,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def dsn():
    value = os.environ.get("SKYDISCOVER_TEST_POSTGRES_DSN")
    if not value:
        pytest.skip("Set SKYDISCOVER_TEST_POSTGRES_DSN to run PostgreSQL tests")
    migrate(value)
    return value


@pytest.fixture
def databases(dsn):
    opened = []
    runs = set()

    def create(cls=TopKPostgresProgramDatabase, cfg=None, name="topk"):
        cfg = cfg or DatabaseConfig(backend="postgres", postgres_dsn=dsn, random_seed=42)
        db = cls(name, cfg)
        opened.append(db)
        runs.add(db.run_id)
        return db

    yield create
    for db in opened:
        db.close()
    with connect(dsn) as conn:
        for run in runs:
            conn.execute("DELETE FROM skydiscover.runs WHERE id=%s", (run,))


def program(pid, score, iteration=0, parent=None):
    return Program(
        id=pid,
        solution=f"def f(): return {score}",
        metrics={"combined_score": score, "latency": 10 - score},
        iteration_found=iteration,
        parent_id=parent,
        artifacts={"feedback": "measured"},
        metadata={"tags": ["candidate"]},
    )


def normalized(selection):
    p, ctx = selection
    parent = next(iter(p.values())) if isinstance(p, dict) else p
    context = [v for group in ctx.values() for v in group] if isinstance(ctx, dict) else ctx
    return parent.id, [v.id for v in context]


@pytest.mark.parametrize(
    "cls,config_cls,name",
    [
        (TopKPostgresProgramDatabase, DatabaseConfig, "topk"),
        (BestOfNPostgresProgramDatabase, BestOfNDatabaseConfig, "best_of_n"),
        (BeamSearchPostgresProgramDatabase, BeamSearchDatabaseConfig, "beam_search"),
        (ClaudeCodePostgresProgramDatabase, DatabaseConfig, "claude_code"),
        (AdaEvolvePostgresProgramDatabase, AdaEvolveDatabaseConfig, "adaevolve"),
        (
            OpenEvolveNativePostgresProgramDatabase,
            OpenEvolveNativeDatabaseConfig,
            "openevolve_native",
        ),
        (GEPANativePostgresProgramDatabase, GEPANativeDatabaseConfig, "gepa_native"),
        (SearchStrategyPostgresProgramDatabase, DatabaseConfig, "evox_meta"),
    ],
)
def test_state_and_selection_survive_reopen(databases, dsn, cls, config_cls, name):
    cfg = config_cls(backend="postgres", postgres_dsn=dsn, random_seed=42)
    db = databases(cls, cfg, name)
    for i in range(6):
        db.add(program(f"p{i}", i, i, "p0" if i else None), iteration=i)
    db.log_prompt("p5", "generation", {"system": "system", "user": "user"}, ["response"])
    db.complete_iteration(0)
    selection = normalized(db.sample_for_iteration(1, 2))
    assert db._conn.execute(
        "SELECT parent_id FROM skydiscover.attempts WHERE run_id=%s AND iteration=1",
        (db.run_id,),
    ).fetchone() == (selection[0],)
    db.complete_iteration(1)
    state = copy.deepcopy(
        db._conn.execute(
            "SELECT state FROM skydiscover.strategies WHERE run_id=%s AND revision=0", (db.run_id,)
        ).fetchone()[0]
    )
    best = db.get_best_program().id
    run_id = db.run_id
    db.close()
    resume_cfg = copy.copy(cfg)
    resume_cfg.run_id = run_id
    resumed = databases(cls, resume_cfg, name)
    assert resumed.get_best_program().id == best
    assert normalized(resumed.sample_for_iteration(1, 2)) == selection
    assert resumed.get_prompts("p5")["generation"]["responses"] == ["response"]
    assert resumed.get("p5").artifacts == {"feedback": "measured"}
    assert resumed.next_iteration == 2
    assert resumed._conn.execute(
        "SELECT parent_id FROM skydiscover.attempts WHERE run_id=%s AND iteration=1",
        (run_id,),
    ).fetchone() == (selection[0],)
    # Exact RNG and policy state survives reopening, not just best candidate.
    assert (
        resumed._conn.execute(
            "SELECT state FROM skydiscover.strategies WHERE run_id=%s AND revision=0", (run_id,)
        ).fetchone()[0]
        == state
    )


def test_atomic_failure_and_detached_update(databases):
    db = databases()
    db.add(program("seed", 1), iteration=0)
    db.complete_iteration(0)
    with pytest.raises(RuntimeError):
        with db.operation():
            db.add(program("child", 2, 1, "seed"), iteration=1)
            db.log_prompt("child", "generation", {"user": "hello"})
            db.complete_iteration(1)
            raise RuntimeError("simulated crash before commit")
    assert db.get("child") is None
    assert db.get_prompts("child") == {}
    assert db.next_iteration == 1
    assert db.get_best_program().id == "seed"
    detached = db.get("seed")
    detached.metrics["test_score"] = 3
    assert "test_score" not in db.get("seed").metrics
    db.update(detached)
    assert db.get("seed").metrics["test_score"] == 3


def test_run_isolation_and_exclusive_lock(databases, dsn):
    a, b = databases(), databases()
    a.add(program("same-id", 1))
    b.add(program("same-id", 2))
    assert a.get("same-id").metrics["combined_score"] == 1
    assert b.get("same-id").metrics["combined_score"] == 2
    with pytest.raises(RuntimeError, match="already open"):
        databases(
            cfg=DatabaseConfig(
                backend="postgres", postgres_dsn=dsn, random_seed=42, run_id=a.run_id
            )
        )


def test_attempt_parent_must_belong_to_same_run(databases):
    from psycopg.errors import ForeignKeyViolation

    a, b = databases(), databases()
    a.add(program("parent", 1))
    with pytest.raises(ForeignKeyViolation):
        b._conn.execute(
            "INSERT INTO skydiscover.attempts(run_id,iteration,parent_id) VALUES (%s,0,%s)",
            (b.run_id, "parent"),
        )
    b.add(program("parent", 2))
    b._conn.execute(
        "INSERT INTO skydiscover.attempts(run_id,iteration,parent_id) VALUES (%s,0,%s)",
        (b.run_id, "parent"),
    )
    b.complete_iteration(0)
    assert b._conn.execute(
        "SELECT parent_id,completed FROM skydiscover.attempts WHERE run_id=%s AND iteration=0",
        (b.run_id,),
    ).fetchone() == ("parent", True)
    b.complete_iteration(1)
    assert b._conn.execute(
        "SELECT parent_id FROM skydiscover.attempts WHERE run_id=%s AND iteration=1",
        (b.run_id,),
    ).fetchone() == (None,)


def test_parallel_completion_does_not_skip_holes(databases):
    db = databases()
    db.complete_iteration(0)
    db.complete_iteration(3)
    db.complete_iteration(1)
    assert db.next_iteration == 2
    assert db.is_iteration_complete(3)
    db.complete_iteration(2, {"counter": 3})
    assert db.next_iteration == 4
    assert db.get_controller_state() == {"counter": 3}


def test_eviction_keeps_full_history(databases, dsn):
    db = databases(
        OpenEvolveNativePostgresProgramDatabase,
        OpenEvolveNativeDatabaseConfig(
            backend="postgres", postgres_dsn=dsn, num_islands=1, population_size=2
        ),
        "openevolve_native",
    )
    for i in range(8):
        db.add(program(str(i), i, i), iteration=i)
    assert db.count() < 8
    assert db.get("0") is not None
    assert (
        db._conn.execute(
            "SELECT count(*) FROM skydiscover.programs WHERE run_id=%s", (db.run_id,)
        ).fetchone()[0]
        == 8
    )


def test_gepa_rejections_and_ada_paradigms(databases, dsn):
    gepa = databases(
        GEPANativePostgresProgramDatabase,
        GEPANativeDatabaseConfig(backend="postgres", postgres_dsn=dsn),
        "gepa_native",
    )
    gepa.add_rejected(program("rejected", 0))
    assert gepa.get_rejection_history()[0].id == "rejected"
    assert gepa.count() == 0
    ada = databases(
        AdaEvolvePostgresProgramDatabase,
        AdaEvolveDatabaseConfig(backend="postgres", postgres_dsn=dsn),
        "adaevolve",
    )
    ada.set_paradigms([{"idea": "different approach", "description": "try it"}])
    assert ada.has_active_paradigm()
    ada.use_paradigm()
    assert ada.get_current_paradigm() is not None


def test_schema_is_identical_for_every_algorithm(databases):
    db = databases()
    tables = db._conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='skydiscover' ORDER BY table_name"
    ).fetchall()
    assert [r[0] for r in tables] == [
        "attempts",
        "memberships",
        "programs",
        "prompts",
        "runs",
        "schema_version",
        "strategies",
    ]


def test_generated_evox_uses_selected_backend(databases, dsn, tmp_path):
    from skydiscover.optimize.config import EvoxDatabaseConfig
    from skydiscover.optimize.search.in_memory_database import InMemoryProgramDatabase
    from skydiscover.optimize.search.postgres_database import PostgresProgramDatabase
    from skydiscover.optimize.search.registry import create_database
    from skydiscover.optimize.search.utils.discovery_utils import load_database_from_file

    cfg = EvoxDatabaseConfig(backend="postgres", postgres_dsn=dsn, random_seed=42)
    db = create_database("evox", cfg)
    run_id = db.run_id
    try:
        assert isinstance(db, PostgresProgramDatabase)
        assert not isinstance(db, InMemoryProgramDatabase)
        db.add(program("seed", 1), iteration=0)
        db.complete_iteration(0)
        parent, ctx = normalized(db.sample_for_iteration(1, 2))
        assert parent == "seed"
        # A new strategy revision uses the same candidates and prompt table.
        source = (
            open(cfg.database_file_path)
            .read()
            .replace("class EvolvedProgramDatabase", "class EvolvedProgramDatabase")
        )
        source += "\n# strategy revision two\n"
        path = tmp_path / "strategy.py"
        path.write_text(source)
        cls, program_cls = load_database_from_file(str(path), backend="postgres")
        newer = cls("evox", db.strategy_config())
        newer._program_class = program_cls
        newer._open()
        newer.set_strategy_source(source)
        for candidate in db.iter_programs():
            newer.add(candidate, iteration=candidate.iteration_found)
        newer.activate_strategy()
        assert newer._revision == 1
        assert (
            newer._conn.execute(
                "SELECT count(*) FROM skydiscover.programs WHERE run_id=%s", (run_id,)
            ).fetchone()[0]
            == 1
        )
        newer.close()
        cfg.run_id = run_id
        resumed = create_database("evox", cfg)
        try:
            assert resumed._revision == 1
            assert resumed.get_best_program().id == "seed"
        finally:
            resumed.close()
    finally:
        db.close()
        with connect(dsn) as conn:
            conn.execute("DELETE FROM skydiscover.runs WHERE id=%s", (run_id,))


@pytest.mark.parametrize(
    "name,cfg_cls",
    [
        ("topk", DatabaseConfig),
        ("best_of_n", BestOfNDatabaseConfig),
        ("adaevolve", AdaEvolveDatabaseConfig),
        ("gepa_native", GEPANativeDatabaseConfig),
    ],
)
@pytest.mark.asyncio
async def test_runner_uses_postgres_without_checkpoints(dsn, tmp_path, monkeypatch, name, cfg_cls):
    from skydiscover.optimize.config import Config, SearchConfig
    from skydiscover.optimize.llm.base import LLMResponse
    from skydiscover.optimize.llm.llm_pool import LLMPool
    from skydiscover.optimize.runner import Runner
    from skydiscover.optimize.search.default_discovery_controller import DiscoveryController

    async def generate(self, *args, **kwargs):
        return LLMResponse(text="```python\ndef solution():\n    return 2\n```")

    monkeypatch.setattr(LLMPool, "generate", generate)
    seed = tmp_path / "initial.py"
    seed.write_text("def solution():\n    return 1\n")
    evaluator = tmp_path / "evaluator.py"
    evaluator.write_text(
        'import runpy\ndef evaluate(program_path):\n    return {"combined_score": runpy.run_path(program_path)["solution"]()}\n'
    )
    db_cfg = cfg_cls(backend="postgres", postgres_dsn=dsn, random_seed=42)
    if name == "gepa_native":
        db_cfg.use_merge = False
    if name == "adaevolve":
        db_cfg.use_paradigm_breakthrough = False
    cfg = Config(search=SearchConfig(type=name, database=db_cfg), diff_based_generation=False)
    from skydiscover.optimize.config import LLMModelConfig

    model = LLMModelConfig(name="test", init_client=lambda cfg: object())
    cfg.llm.models = [model]
    cfg.llm.evaluator_models = [model]
    cfg.llm.guide_models = [model]
    cfg.evaluator.cascade_evaluation = False
    runner = Runner(str(evaluator), str(seed), config=cfg, output_dir=str(tmp_path / "first"))
    run_id = runner.run_id
    try:
        best = await runner.run(iterations=2)
        assert best.metrics["combined_score"] == 2
        assert best.metrics["test_combined_score"] == 2
        assert not (tmp_path / "first" / "checkpoints").exists()
        assert runner.initial_score == 1
        resumed = Runner(
            str(evaluator),
            str(seed),
            config=cfg,
            output_dir=str(tmp_path / "resumed"),
            resume=run_id,
        )
        assert cfg.search.database.run_id is None
        assert resumed.database.next_iteration == 3
        assert resumed.database.get_best_program().metrics["test_combined_score"] == 2
        best = await resumed.run(iterations=1)
        assert best.metrics["combined_score"] == 2
        assert not (tmp_path / "resumed" / "checkpoints").exists()
    finally:
        runner.database.close()
        with connect(dsn) as conn:
            conn.execute("DELETE FROM skydiscover.runs WHERE id=%s", (run_id,))


def test_nonfinite_metrics_and_binary_artifacts(databases):
    import math

    db = databases()
    p = program("edge", 1)
    p.metrics["invalid"] = float("nan")
    p.artifacts["binary"] = b"\x00\x01"
    p.parent_info = ("parent label", "missing")
    db.add(p)
    actual = db.get("edge")
    assert math.isnan(actual.metrics["invalid"])
    assert actual.artifacts["binary"] == b"\x00\x01"
    assert actual.parent_info == ("parent label", "missing")


def test_process_death_rolls_back_and_releases_run_lock(databases, dsn):
    import subprocess
    import sys

    child = """
import os
from skydiscover.optimize.config import DatabaseConfig
from skydiscover.optimize.search.postgres_algorithms import TopKPostgresProgramDatabase
from skydiscover.optimize.search.base_database import Program
config = DatabaseConfig(backend="postgres", postgres_dsn=os.environ["SKYDISCOVER_TEST_POSTGRES_DSN"], random_seed=42)
db = TopKPostgresProgramDatabase("topk", config)
db.add(Program(id="seed", solution="seed", metrics={"combined_score":1}))
db.complete_iteration(0)
db.sample_for_iteration(1, 2)
with db.operation():
    db.add(Program(id="uncommitted", solution="child", iteration_found=1, metrics={"combined_score":9}), iteration=1)
    db.complete_iteration(1)
    print(db.run_id, flush=True)
    os._exit(17)
"""
    result = subprocess.run(
        [sys.executable, "-c", child], capture_output=True, text=True, timeout=20
    )
    assert result.returncode == 17, result.stderr
    run_id = result.stdout.strip()
    db = databases(
        cfg=DatabaseConfig(backend="postgres", postgres_dsn=dsn, run_id=run_id, random_seed=42)
    )
    assert db.next_iteration == 1
    assert db.get("uncommitted") is None
    assert db.get_best_program().id == "seed"
    assert normalized(db.sample_for_iteration(1, 2))[0] == "seed"


def test_viewer_can_read_an_active_run(databases, dsn):
    from skydiscover.optimize.extras.monitor.viewer import load_postgres_run

    db = databases()
    db.add(program("seed", 1))
    db.complete_iteration(0)
    db.log_prompt("seed", "generation", {"user": "hello"}, ["result"])
    programs, best_id, iteration = load_postgres_run(dsn, db.run_id)
    assert programs[0]["prompts"]["generation"]["responses"] == ["result"]
    assert best_id == "seed"
    assert iteration == 0


@pytest.mark.parametrize(
    "name,cfg_cls,pg_cls",
    [
        ("topk", DatabaseConfig, TopKPostgresProgramDatabase),
        ("best_of_n", BestOfNDatabaseConfig, BestOfNPostgresProgramDatabase),
        ("beam_search", BeamSearchDatabaseConfig, BeamSearchPostgresProgramDatabase),
        ("adaevolve", AdaEvolveDatabaseConfig, AdaEvolvePostgresProgramDatabase),
        (
            "openevolve_native",
            OpenEvolveNativeDatabaseConfig,
            OpenEvolveNativePostgresProgramDatabase,
        ),
        ("gepa_native", GEPANativeDatabaseConfig, GEPANativePostgresProgramDatabase),
    ],
)
def test_sampling_matches_memory_and_rng_continues(databases, dsn, name, cfg_cls, pg_cls):
    from skydiscover.optimize.search import route  # register memory algorithms
    from skydiscover.optimize.search.registry import create_database

    memory_cfg = cfg_cls(random_seed=42)
    pg_cfg = cfg_cls(backend="postgres", postgres_dsn=dsn, random_seed=42)
    memory = create_database(name, memory_cfg)
    pg = databases(pg_cls, pg_cfg, name)
    for i in range(5):
        candidate = program(str(i), i, i)
        memory.add(copy.deepcopy(candidate), iteration=i)
        pg.add(copy.deepcopy(candidate), iteration=i)
    for iteration in range(1, 4):
        assert normalized(pg.sample_for_iteration(iteration, 2)) == normalized(
            memory.sample_for_iteration(iteration, 2)
        )
    run_id = pg.run_id
    pg.close()
    pg_cfg.run_id = run_id
    pg = databases(pg_cls, pg_cfg, name)
    assert normalized(pg.sample_for_iteration(4, 2)) == normalized(
        memory.sample_for_iteration(4, 2)
    )
    with pytest.raises(ValueError, match="run IDs"):
        pg.save("ignored")


def test_claude_snapshots_persist_turn_progress(databases):
    from skydiscover.optimize.search.claude_code.controller import ClaudeCodeController

    db = databases(ClaudeCodePostgresProgramDatabase, name="claude_code")
    db.complete_iteration(0)
    controller = ClaudeCodeController.__new__(ClaudeCodeController)
    controller.database = db
    controller.early_stopping_triggered = False
    controller._commit_snapshot(program("snapshot", 3, 7), 7)
    assert db.next_iteration == 8
    assert db.is_iteration_complete(6)
    assert db.get_best_program().id == "snapshot"


@pytest.mark.asyncio
async def test_evox_controller_resume_preserves_meta_run_and_window(dsn, tmp_path, monkeypatch):
    from skydiscover.optimize.config import Config, EvoxDatabaseConfig, LLMModelConfig, SearchConfig
    from skydiscover.optimize.llm.base import LLMResponse
    from skydiscover.optimize.llm.llm_pool import LLMPool
    from skydiscover.optimize.runner import Runner
    from skydiscover.optimize.search.evox.controller import CoEvolutionController

    async def generate(self, *args, **kwargs):
        return LLMResponse(text="```python\ndef solution():\n    return 2\n```")

    async def availability(self):
        return True

    async def labels(self):
        self._diverge_label = "explore"
        self._refine_label = "refine"
        self._assign_labels_to_db(self.database)

    monkeypatch.setattr(LLMPool, "generate", generate)
    monkeypatch.setattr(LLMPool, "check_availability", availability)
    monkeypatch.setattr(CoEvolutionController, "_generate_variation_operators", labels)
    seed = tmp_path / "initial.py"
    seed.write_text("def solution():\n    return 1\n")
    evaluator = tmp_path / "evaluator.py"
    evaluator.write_text(
        'import runpy\ndef evaluate(program_path):\n    return {"combined_score": runpy.run_path(program_path)["solution"]()}\n'
    )
    db_cfg = EvoxDatabaseConfig(backend="postgres", postgres_dsn=dsn, random_seed=42)
    cfg = Config(
        search=SearchConfig(type="evox", database=db_cfg, switch_interval=100, share_llm=True),
        diff_based_generation=False,
    )
    model = LLMModelConfig(name="test", init_client=lambda c: object())
    cfg.llm.models = cfg.llm.evaluator_models = cfg.llm.guide_models = [model]
    cfg.evaluator.cascade_evaluation = False
    runner = Runner(str(evaluator), str(seed), config=cfg, output_dir=str(tmp_path / "first"))
    run_id = runner.run_id
    meta_run = None
    try:
        best = await runner.run(iterations=2)
        assert best.metrics["combined_score"] == 2
        resumed = Runner(
            str(evaluator),
            str(seed),
            config=cfg,
            output_dir=str(tmp_path / "second"),
            resume=run_id,
        )
        state = resumed.database.get_controller_state()
        meta_run = state["meta_run_id"]
        assert state["search_scorer"]["_start_score"] == 1
        assert len(state["search_scorer"]["_best_scores"]) == 2
        assert state["_diverge_label"] == "explore"
        await resumed.run(iterations=1)
        from skydiscover.optimize.search.registry import create_database

        resume_cfg = copy.copy(db_cfg)
        resume_cfg.run_id = run_id
        reopened = create_database("evox", resume_cfg)
        try:
            state = reopened.get_controller_state()
            assert state["meta_run_id"] == meta_run
            assert len(state["search_scorer"]["_best_scores"]) == 3
        finally:
            reopened.close()
    finally:
        runner.database.close()
        with connect(dsn) as conn:
            conn.execute("DELETE FROM skydiscover.runs WHERE id=%s", (run_id,))
            if meta_run:
                conn.execute("DELETE FROM skydiscover.runs WHERE id=%s", (meta_run,))


def test_terminal_failure_keeps_prompt_and_response(databases):
    from skydiscover.optimize.search.default_discovery_controller import DiscoveryController
    from skydiscover.optimize.search.utils.discovery_utils import SerializableResult

    db = databases()
    controller = DiscoveryController.__new__(DiscoveryController)
    controller.database = db
    controller.early_stopping_triggered = False
    result = SerializableResult(
        iteration=0, error="invalid response", prompt={"user": "task"}, llm_response="unparseable"
    )
    controller._complete_iteration(0, result.error, result)
    stored = db._conn.execute(
        "SELECT outcome FROM skydiscover.attempts WHERE run_id=%s AND iteration=0", (db.run_id,)
    ).fetchone()[0]
    assert stored["llm_response"] == "unparseable"
    assert stored["prompt"] == {"user": "task"}
    assert db.next_iteration == 1


def test_boolean_score_retains_its_value_and_can_be_ranked(databases):
    db = databases()
    db.add(program("correct", True))
    assert db.get("correct").metrics["combined_score"] is True
    assert db.get_top_programs(1)[0].id == "correct"
