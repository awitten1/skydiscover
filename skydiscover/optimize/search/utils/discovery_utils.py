import importlib.util
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

from skydiscover.optimize.search.base_database import Program, ProgramDatabase

logger = logging.getLogger(__name__)


def load_evaluator_code(evaluation_file: Optional[str]) -> str:
    """Return evaluator source as a string for LLM context.

    For a plain Python file, returns its contents.
    For a containerized benchmark directory, returns all text files except
    infrastructure files (Dockerfile, requirements.txt) and data files (.json).
    """
    if not evaluation_file:
        return ""
    try:
        p = Path(evaluation_file)
        if not p.exists():
            return ""
        if p.is_dir():
            # Harbor task: prioritize instruction.md — it contains the full
            # problem description, reference implementation, and constraints.
            instruction = p / "instruction.md"
            if instruction.exists():
                return instruction.read_text()

            _SKIP = {"Dockerfile", "requirements.txt"}
            _MAX_FILES = 10
            _MAX_BYTES = 50_000
            parts = []
            for f in sorted(p.iterdir()):
                if len(parts) >= _MAX_FILES:
                    break
                if not f.is_file():
                    continue
                if f.name in _SKIP or f.suffix == ".json":
                    continue
                if f.stat().st_size > _MAX_BYTES:
                    continue
                try:
                    parts.append(f"# {f.name}\n{f.read_text()}")
                except Exception:
                    pass  # skip binary or unreadable files
            return "\n\n".join(parts)
        return p.read_text()
    except Exception as e:
        logger.warning(f"Could not load evaluator code from {evaluation_file}: {e}")
        return ""


@dataclass
class SerializableResult:
    """Result that can be pickled and sent between processes"""

    child_program_dict: Optional[Dict[str, Any]] = None
    parent_id: Optional[str] = None
    other_context_ids: Optional[List[str]] = None

    iteration_time: float = 0.0
    llm_generation_time: float = 0.0
    eval_time: float = 0.0
    prompt: Optional[Dict[str, str]] = None
    llm_response: Optional[str] = None
    iteration: int = 0
    error: Optional[str] = None
    attempts_used: int = 1


def load_database_from_file(
    file_path: str,
    database_class_name: str = "EvolvedProgramDatabase",
    program_class_name: str = "EvolvedProgram",
    backend: str = "memory",
) -> Tuple[Type[ProgramDatabase], Type[Program]]:
    """Dynamically load database and program classes from a Python file."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Database file not found: {file_path}")

    import hashlib

    source = Path(file_path).read_text()
    if backend not in ("memory", "postgres"):
        raise ValueError(f"Unknown database backend: {backend}")
    module_name = "custom_database_" + hashlib.sha256((source + backend).encode()).hexdigest()[:16]

    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(module_name, file_path)

        if spec is None or spec.loader is None:
            raise ValueError(f"Could not load module from: {file_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module

        try:
            # Generated strategies inherit the selected storage backend. Keep
            # their Program import and method bodies unchanged, including super().
            import ast

            tree = ast.parse(source, filename=file_path)

            class BackendImports(ast.NodeTransformer):
                def visit_ImportFrom(self, node):
                    if node.module not in (
                        "skydiscover.optimize.search.base_database",
                        "skydiscover.search.base_database",
                        "skydiscover.optimize.search.in_memory_database",
                        "skydiscover.optimize.search.postgres_database",
                    ):
                        return node
                    storage = [
                        n
                        for n in node.names
                        if n.name
                        in ("ProgramDatabase", "InMemoryProgramDatabase", "PostgresProgramDatabase")
                    ]
                    if not storage:
                        return node
                    other = [n for n in node.names if n not in storage]
                    target_module = f"skydiscover.optimize.search.{ 'postgres_database' if backend == 'postgres' else 'in_memory_database'}"
                    target_name = (
                        "PostgresProgramDatabase"
                        if backend == "postgres"
                        else "InMemoryProgramDatabase"
                    )
                    imports = [
                        ast.ImportFrom(
                            module=target_module,
                            names=[ast.alias(name=target_name, asname=n.asname or n.name)],
                            level=0,
                        )
                        for n in storage
                    ]
                    if other:
                        imports.insert(
                            0, ast.ImportFrom(module=node.module, names=other, level=node.level)
                        )
                    return imports

            tree = ast.fix_missing_locations(BackendImports().visit(tree))
            exec(compile(tree, file_path, "exec"), module.__dict__)
        except Exception as e:
            del sys.modules[module_name]
            raise ValueError(f"Error executing {file_path}: {e}") from e

    module = sys.modules[module_name]
    database_class = getattr(module, database_class_name, None)
    program_class = getattr(module, program_class_name, None)

    if database_class is None or program_class is None:
        raise AttributeError(
            f"Expected {database_class_name} and {program_class_name} in {file_path}"
        )

    if not issubclass(database_class, ProgramDatabase) or not issubclass(program_class, Program):
        raise TypeError(
            f"{database_class_name} must extend ProgramDatabase, {program_class_name} must extend Program"
        )

    from skydiscover.optimize.search.persistence.operations import database_operation

    # Generated public methods follow the same transaction boundary as native ones.
    for name, method in list(vars(database_class).items()):
        if callable(method) and not name.startswith("_") and name not in ("save", "load"):
            setattr(database_class, name, database_operation(method))
    return database_class, program_class


def build_image_content(text_prompt: str, parent: Program, other_context: dict) -> list:
    """Build multimodal content array with images for VLM (image generation mode).

    Encodes parent and other context images as base64 and interleaves them
    with the text prompt so the VLM can see what the current images look like.
    """
    import base64
    import os

    _MIME = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
        "gif": "image/gif",
    }

    def _encode_image(path: str) -> dict | None:
        if not path or not os.path.exists(path):
            return None
        try:
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            ext = os.path.splitext(path)[1].lstrip(".").lower()
            mime = _MIME.get(ext, "image/png")
            return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
        except Exception:
            return None

    content = []

    # Parent image
    parent_img = (getattr(parent, "metadata", {}) or {}).get("image_path")
    img_part = _encode_image(parent_img)
    if img_part:
        score = (parent.metrics or {}).get("combined_score", "?")
        content.append({"type": "text", "text": f"Current best image (score: {score}):"})
        content.append(img_part)

    # Other context images (limit to 3 to keep token cost reasonable)
    img_count = 0
    for progs in other_context.values():
        for prog in progs:
            if img_count >= 3:
                break
            prog_img = (getattr(prog, "metadata", {}) or {}).get("image_path")
            img_part = _encode_image(prog_img)
            if img_part:
                score = (prog.metrics or {}).get("combined_score", "?")
                content.append({"type": "text", "text": f"Other context images (score: {score}):"})
                content.append(img_part)
                img_count += 1

    # Text prompt (with all the formatted context from prompt generator)
    content.append({"type": "text", "text": text_prompt})

    return content
