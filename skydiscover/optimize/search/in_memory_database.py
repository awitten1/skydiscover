"""In-memory backend. Legacy file helpers remain for existing callers."""

import os
import random
from typing import Any, Dict, List, Optional

from skydiscover.optimize.config import DatabaseConfig
from skydiscover.optimize.search.base_database import Program, ProgramDatabase


class InMemoryProgramDatabase(ProgramDatabase):
    def __init__(self, name: str, config: DatabaseConfig, **kwargs: Any):
        self.name = name
        self.config = config

        # In-memory program storage
        # Per-database RNG. Seeding the module-global `random` would reach every
        # other component in the process; this keeps selection reproducible without
        # that side effect. random.Random(None) draws from OS entropy, so an unset
        # seed behaves exactly as before.
        self.rng: random.Random = random.Random(getattr(config, "random_seed", None))

        self.programs: Dict[str, Program] = {}
        # Set by Runner from the resolved config; subclasses read it when
        # rendering prompts and naming saved program files.
        self.language: str = "python"

        # Track the last iteration number (for resuming)
        self.last_iteration: int = 0

        # Optionally track initial program info (set by controller on first add)
        self.initial_program_id: Optional[str] = None
        self.initial_program_score: Optional[float] = None

        # Best program tracking
        self.best_program_id: Optional[str] = None

        # Lazy Pareto-front cache (invalidated on add when multiobjective)
        self._pareto_front_cache: Optional[List[Program]] = None
        self._pareto_front_cache_valid: bool = False

        # Prompt log
        self.prompts_by_program: Optional[Dict[str, Dict[str, Any]]] = None

        # Initialize checkpoint manager (imported here to avoid circular imports)
        from skydiscover.optimize.search.utils.checkpoint_manager import CheckpointManager

        self.checkpoint_manager = CheckpointManager(self.config)

        # Load database from disk if path is provided
        if config.db_path and os.path.exists(config.db_path):
            self.load(config.db_path)

    def get(self, program_id):
        return self.programs.get(program_id)

    def count(self):
        return len(self.programs)

    # ------------------------------------------------------------------
    # Save and load
    # ------------------------------------------------------------------
    def save(
        self,
        path: Optional[str] = None,
        iteration: int = 0,
        programs: Optional[Dict[str, Program]] = None,
    ) -> None:
        """
        Save the database to disk

        Args:
            path: Path to save to (uses config.db_path if None)
            iteration: Current iteration number
            programs: Snapshot to write instead of ``self.programs``. Subclasses
                pass one when the checkpoint should hold something other than the
                live registry; saving must never mutate in-memory state.
        """
        self.checkpoint_manager.save(
            programs=self.programs if programs is None else programs,
            prompts_by_program=self.prompts_by_program,
            best_program_id=self.best_program_id,
            last_iteration=iteration if iteration is not None else self.last_iteration,
            path=path,
        )

    def load(self, path: str) -> None:
        """
        Load the database from disk

        Args:
            path: Path to load from
        """
        programs, best_id, last_iter = self.checkpoint_manager.load(path)
        self.programs = programs
        self.best_program_id = best_id
        self.last_iteration = last_iter

        self.log_status()

    def _save_program(
        self,
        program: Program,
        base_path: Optional[str] = None,
        prompts: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> None:
        """
        Save a single program to disk.

        This is a convenience method that delegates to CheckpointManager.
        Subclasses should use this method when they need to save individual programs
        (e.g., during add() operations).

        Args:
            program: Program to save
            base_path: Base path to save to (uses config.db_path if None)
            prompts: Optional prompts to save with the program
        """
        self.checkpoint_manager._save_program(program, base_path, prompts)
