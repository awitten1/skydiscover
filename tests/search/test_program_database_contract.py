"""Storage hierarchy and factory contract, without an external database."""

import pytest

from skydiscover.optimize.config import DatabaseConfig
from skydiscover.optimize.search.base_database import ProgramDatabase
from skydiscover.optimize.search.best_of_n.database import InMemoryBestOfNProgramDatabase
from skydiscover.optimize.search.in_memory_database import InMemoryProgramDatabase
from skydiscover.optimize.search.postgres_algorithms import BestOfNPostgresProgramDatabase
from skydiscover.optimize.search.postgres_database import PostgresProgramDatabase
from skydiscover.optimize.search.registry import create_database, register_database


def test_abc_does_not_allocate_backend_storage():
    class ContractOnly(ProgramDatabase):
        def add(self, program, iteration=None, **kwargs):
            return program.id

        def sample(self, num_context_programs=4, **kwargs):
            return None, []

        def get(self, program_id):
            return None

        def count(self):
            return 0

    db = ContractOnly("contract", DatabaseConfig())
    assert set(vars(db)) == {"name", "config"}
    assert not db.has_programs()


def test_algorithm_backends_are_distinct_branches():
    assert issubclass(InMemoryBestOfNProgramDatabase, InMemoryProgramDatabase)
    assert issubclass(BestOfNPostgresProgramDatabase, PostgresProgramDatabase)
    assert not issubclass(BestOfNPostgresProgramDatabase, InMemoryProgramDatabase)
    with pytest.raises(TypeError, match="must inherit"):
        register_database("invalid", InMemoryBestOfNProgramDatabase, backend="postgres")


def test_invalid_backend_and_memory_resume_fail_before_opening_storage():
    with pytest.raises(ValueError, match="Unknown database backend"):
        create_database("best_of_n", DatabaseConfig(backend="invalid"))
    with pytest.raises(ValueError, match="requires the PostgreSQL"):
        create_database("best_of_n", DatabaseConfig(run_id="example"))
    with pytest.raises(ValueError, match="Tasks require the PostgreSQL"):
        create_database("best_of_n", DatabaseConfig(task_id="example"))


def test_operation_wrapping_is_idempotent_for_cached_evox_classes():
    from skydiscover.optimize.search.persistence.operations import database_operation

    def method(self):
        return 1

    wrapped = database_operation(method)
    assert database_operation(wrapped) is wrapped


def test_full_configuration_snapshot_roundtrips_all_sections():
    import copy

    from skydiscover.optimize.config import Config, LLMModelConfig
    from skydiscover.optimize.search.persistence.inputs import configuration_snapshot

    cfg = Config(language="rust", file_suffix=".rs", max_solution_length=12345)
    cfg.search.switch_interval = 19
    cfg.context_builder.suggest_simplification_after_chars = 234
    cfg.benchmark.params = {"dataset": "task-data", "limit": 7}
    cfg.llm.models = [LLMModelConfig(name="test", weight=0.7, api_key="private")]
    cfg.llm.guide_models = [LLMModelConfig(name="guide", reasoning_effort="high")]
    cfg.search.database.extra_setting = {"value": 3}
    data = configuration_snapshot(cfg)
    restored = Config.from_dict(copy.deepcopy(data))
    assert restored.language == "rust"
    assert restored.file_suffix == ".rs"
    assert restored.max_solution_length == 12345
    assert restored.search.switch_interval == 19
    assert restored.context_builder.suggest_simplification_after_chars == 234
    assert restored.benchmark.params == cfg.benchmark.params
    assert restored.llm.models[0].weight == 0.7
    assert restored.llm.models[0].api_key is None
    assert restored.llm.guide_models[0].reasoning_effort == "high"
    assert restored.search.database.extra_setting == {"value": 3}


@pytest.mark.parametrize("path", ["../escape.py", "/tmp/escape.py", "nested/../../escape.py"])
def test_stored_asset_paths_cannot_escape_workspace(tmp_path, path):
    from skydiscover.optimize.config import Config
    from skydiscover.optimize.search.persistence.inputs import materialize_inputs

    payload = {"assets": {"files": {path: {"data": "", "mode": 0o644}}}}
    with pytest.raises(ValueError, match="Invalid stored input path"):
        materialize_inputs(payload, str(tmp_path), Config())
