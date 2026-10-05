"""Durable configuration and input files for self-contained PostgreSQL runs."""

import base64
import copy
import json
import os
import uuid
from dataclasses import asdict
from pathlib import Path, PurePosixPath

from skydiscover.optimize.config import Config, resolve_config
from skydiscover.optimize.search.persistence.schema import connect

_PREFIX = "assets:/"
_EXCLUDED = {".git", ".venv", "__pycache__", ".pytest_cache", "node_modules", ".next"}


def configuration_snapshot(config):
    """Capture all declared fields, retaining data rather than runtime clients."""
    data = asdict(config)
    data["prompt"] = data.pop("context_builder")
    params = data["benchmark"].pop("params")
    data["benchmark"].update(params)
    for model in [data["llm"]] + [
        model
        for field in ("models", "evaluator_models", "guide_models")
        for model in data["llm"][field]
    ]:
        model["api_key"] = None
        model["init_client"] = None
    database = data["search"]["database"]
    for key in ("postgres_dsn", "run_id", "task_id", "task_name"):
        database.pop(key, None)
    # Generated strategy configuration can have additional public settings.
    for key, value in vars(config.search.database).items():
        if (
            key not in database
            and not key.startswith("_")
            and key not in {"postgres_dsn", "run_id", "task_id", "task_name"}
        ):
            database[key] = value
    json.dumps(data, allow_nan=False)
    return data


def capture_inputs(config, evaluation_file, output_dir, evaluator_env_vars=None):
    data = configuration_snapshot(config)
    assets = {"files": {}, "directories": [], "environment": {}}
    excluded_output = Path(output_dir).resolve()

    def capture(source, label):
        path = Path(source).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Cannot persist missing input: {path}")
        root = path if path.is_dir() else path.parent

        def visit(directory):
            assets["directories"].append(f"{label}/{directory.relative_to(root).as_posix()}")
            for item in sorted(directory.iterdir()):
                if item.name in _EXCLUDED or item.name.startswith(".env"):
                    continue
                if item.resolve() == excluded_output:
                    continue
                if item.is_dir():
                    if (item / "run_id.json").exists():
                        continue
                    if item.is_symlink():
                        raise ValueError(f"Input directory symlinks are unsupported: {item}")
                    visit(item)
                elif item.is_file():
                    relative = f"{label}/{item.relative_to(root).as_posix()}"
                    assets["files"][relative] = {
                        "data": base64.b64encode(item.read_bytes()).decode("ascii"),
                        "mode": item.stat().st_mode & 0o777,
                    }

        visit(root)
        return _PREFIX + label + ("/" + path.name if path.is_file() else "")

    # Evaluators are explicitly external; remember their location as a default.
    data["evaluator"]["evaluation_file"] = str(Path(evaluation_file).resolve())
    paths = [
        (data["prompt"], "template_dir", "templates"),
        (data["agentic"], "codebase_root", "codebase"),
        (data["search"]["database"], "database_file_path", "strategy"),
        (data["search"]["database"], "evaluation_file", "strategy_evaluator"),
        (data["search"]["database"], "config_path", "strategy_config"),
    ]
    for section, field, label in paths:
        if section.get(field):
            section[field] = capture(section[field], label)
    for key, value in (evaluator_env_vars or {}).items():
        if any(part in key.upper() for part in ("API_KEY", "TOKEN", "PASSWORD", "SECRET")):
            assets["environment"][key] = {"env": key}
        else:
            assets["environment"][key] = value
    return data, assets


def read_inputs(dsn, *, run_id=None, task_id=None):
    if bool(run_id) == bool(task_id):
        raise ValueError("Specify exactly one run ID or task ID")
    if not dsn:
        raise ValueError("Set SKYDISCOVER_POSTGRES_DSN or supply postgres_dsn")
    identity = str(uuid.UUID(run_id or task_id))
    with connect(dsn) as conn:
        if run_id:
            row = conn.execute(
                "SELECT config,assets,starting_solution,starting_filename,task_id "
                "FROM skydiscover.runs WHERE id=%s",
                (identity,),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT config,assets,NULL,NULL,id FROM skydiscover.tasks WHERE id=%s",
                (identity,),
            ).fetchone()
    if not row:
        raise ValueError(f"Unknown {'run' if run_id else 'task'} ID {identity}")
    if not row[0] or not row[1]:
        raise ValueError("Stored inputs are missing; create the task through an Optimize runner")
    return {
        "config": row[0],
        "assets": row[1],
        "starting_solution": row[2],
        "starting_filename": row[3],
        "task_id": str(row[4]),
    }


def restore_configuration(payload, dsn):
    config = resolve_config(Config.from_dict(copy.deepcopy(payload["config"])))
    config.search.database.backend = "postgres"
    config.search.database.postgres_dsn = dsn
    return config


def materialize_inputs(payload, directory, config):
    root = Path(directory).resolve()

    def target(relative):
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts or "\\" in relative:
            raise ValueError(f"Invalid stored input path: {relative}")
        result = root.joinpath(*path.parts)
        result.parent.mkdir(parents=True, exist_ok=True)
        return result

    for relative in payload["assets"].get("directories", []):
        target(relative).mkdir(parents=True, exist_ok=True)
    for relative, entry in payload["assets"]["files"].items():
        path = target(relative)
        path.write_bytes(base64.b64decode(entry["data"], validate=True))
        path.chmod(entry["mode"] & 0o777)

    def resolve(value):
        return str(target(value[len(_PREFIX) :])) if value.startswith(_PREFIX) else value

    stored = payload["config"]
    for section, field, original in [
        (config.evaluator, "evaluation_file", stored["evaluator"].get("evaluation_file")),
        (config.context_builder, "template_dir", stored["prompt"].get("template_dir")),
        (config.agentic, "codebase_root", stored["agentic"].get("codebase_root")),
        *[
            (config.search.database, field, stored["search"]["database"].get(field))
            for field in ("database_file_path", "evaluation_file", "config_path")
        ],
    ]:
        if original:
            setattr(section, field, resolve(original))
    environment = {}
    for key, value in payload["assets"].get("environment", {}).items():
        if isinstance(value, dict):
            environment[key] = os.environ[value["env"]]
        else:
            environment[key] = resolve(value)
    return config.evaluator.evaluation_file, environment
