"""Algorithm subclasses. All use PostgresProgramDatabase's schema and transactions."""

from skydiscover.optimize.search.adaevolve.database import AdaEvolveDatabaseMethods
from skydiscover.optimize.search.beam_search.database import BeamSearchDatabaseMethods
from skydiscover.optimize.search.best_of_n.database import BestOfNDatabaseMethods
from skydiscover.optimize.search.claude_code.database import ClaudeCodeDatabaseMethods
from skydiscover.optimize.search.evox.database.search_strategy_db import (
    SearchStrategyDatabaseMethods,
)
from skydiscover.optimize.search.gepa_native.database import GEPANativeDatabaseMethods
from skydiscover.optimize.search.openevolve_native.database import OpenEvolveNativeDatabaseMethods
from skydiscover.optimize.search.persistence.operations import database_operation
from skydiscover.optimize.search.postgres_database import PostgresProgramDatabase
from skydiscover.optimize.search.topk.database import TopKDatabaseMethods


class BestOfNPostgresProgramDatabase(BestOfNDatabaseMethods, PostgresProgramDatabase):
    save = PostgresProgramDatabase.save
    load = PostgresProgramDatabase.load

    def __init__(self, name, config):
        super().__init__(name, config)
        self._open()

    @database_operation
    def sample(self, num_context_programs=4, **kwargs):
        if not self.has_programs():
            raise ValueError("Cannot sample: no programs in database")
        if (
            self.current_parent_id is None
            or self.parent_iteration_count >= self.n
            or self.current_parent_id not in self.programs
        ):
            row = self._conn.execute(
                "SELECT p.id FROM skydiscover.programs p JOIN skydiscover.memberships m "
                "ON (p.run_id=m.run_id AND p.id=m.program_id) WHERE p.run_id=%s AND m.revision=%s AND m.active "
                "ORDER BY coalesce(p.combined_score, '-Infinity'::float8) DESC,p.ordinal LIMIT 1",
                (self.run_id, self._revision),
            ).fetchone()
            self.current_parent_id = row[0]
            self.parent_iteration_count = 0
        parent = self.get(self.current_parent_id)
        pool = [
            p for p in self.get_top_programs(max(num_context_programs * 2, 10)) if p.id != parent.id
        ]
        return parent, self.rng.sample(pool, min(num_context_programs, len(pool)))


class TopKPostgresProgramDatabase(TopKDatabaseMethods, PostgresProgramDatabase):
    save = PostgresProgramDatabase.save
    load = PostgresProgramDatabase.load

    def __init__(self, name, config):
        super().__init__(name, config)
        self._open()


class BeamSearchPostgresProgramDatabase(BeamSearchDatabaseMethods, PostgresProgramDatabase):
    save = PostgresProgramDatabase.save
    load = PostgresProgramDatabase.load

    def __init__(self, name, config):
        super().__init__(name, config)
        self._open()


class ClaudeCodePostgresProgramDatabase(ClaudeCodeDatabaseMethods, PostgresProgramDatabase):
    save = PostgresProgramDatabase.save
    load = PostgresProgramDatabase.load

    def __init__(self, name, config):
        super().__init__(name, config)
        self._open()


class GEPANativePostgresProgramDatabase(GEPANativeDatabaseMethods, PostgresProgramDatabase):
    save = PostgresProgramDatabase.save
    load = PostgresProgramDatabase.load

    def __init__(self, name, config):
        super().__init__(name, config)
        self._open()


class OpenEvolveNativePostgresProgramDatabase(
    OpenEvolveNativeDatabaseMethods, PostgresProgramDatabase
):
    save = PostgresProgramDatabase.save
    load = PostgresProgramDatabase.load

    def __init__(self, name, config):
        super().__init__(name, config)
        self._open()


class AdaEvolvePostgresProgramDatabase(AdaEvolveDatabaseMethods, PostgresProgramDatabase):
    save = PostgresProgramDatabase.save
    load = PostgresProgramDatabase.load

    def __init__(self, name, config):
        super().__init__(name, config)
        self._open()


class SearchStrategyPostgresProgramDatabase(SearchStrategyDatabaseMethods, PostgresProgramDatabase):
    save = PostgresProgramDatabase.save
    load = PostgresProgramDatabase.load

    def __init__(self, name, config):
        super().__init__(name, config)
        self._open()

    # SearchStrategy needs no extra state beyond the shared backend.
