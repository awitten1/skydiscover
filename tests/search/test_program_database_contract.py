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


def test_operation_wrapping_is_idempotent_for_cached_evox_classes():
    from skydiscover.optimize.search.persistence.operations import database_operation

    def method(self):
        return 1

    wrapped = database_operation(method)
    assert database_operation(wrapped) is wrapped
