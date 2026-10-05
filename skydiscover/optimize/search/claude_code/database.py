"""Minimal database for the Claude Code baseline.

Claude Code handles its own internal iteration loop, so this database
just stores whatever the controller adds (typically one final result).
"""

from skydiscover.optimize.search.base_database import Program, ProgramDatabase
from skydiscover.optimize.search.in_memory_database import InMemoryProgramDatabase
from skydiscover.optimize.search.persistence.operations import database_operation


class ClaudeCodeDatabaseMethods(ProgramDatabase):
    @database_operation
    def add(self, program: Program, iteration=None, **kwargs) -> str:
        self.programs[program.id] = program
        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)
        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)
        return program.id

    @database_operation
    def sample(self, num_context_programs=4, **kwargs):
        best = self.get_best_program()
        return best, []


class InMemoryClaudeCodeProgramDatabase(ClaudeCodeDatabaseMethods, InMemoryProgramDatabase):
    """ClaudeCode selection with in-memory storage."""

    pass


# Preserve the existing import path.
ClaudeCodeDatabase = InMemoryClaudeCodeProgramDatabase
