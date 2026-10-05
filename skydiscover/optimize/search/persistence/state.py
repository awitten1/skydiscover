"""Versioned JSON state codec. No pickle, dynamic imports, or executable deserialization.

Candidates are references to the shared programs table. The object allowlist is
limited to the native algorithms' data structures. Backend transaction caches are omitted.
"""

import math
import random
from collections import defaultdict, deque

from skydiscover.optimize.search.base_database import Program


def _types():
    from skydiscover.optimize.search.adaevolve.adaptation import (
        AdaptiveState,
        MultiDimensionalAdapter,
    )
    from skydiscover.optimize.search.adaevolve.archive.diversity import (
        CodeDiversity,
        HybridDiversity,
        MetricDiversity,
    )
    from skydiscover.optimize.search.adaevolve.archive.unified_archive import (
        ArchiveConfig,
        UnifiedArchive,
    )
    from skydiscover.optimize.search.adaevolve.paradigm.tracker import ParadigmTracker

    return {
        c.__name__: c
        for c in (
            AdaptiveState,
            MultiDimensionalAdapter,
            ArchiveConfig,
            UnifiedArchive,
            CodeDiversity,
            MetricDiversity,
            HybridDiversity,
            ParadigmTracker,
        )
    }


def encode(value, db=None):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"type": "float", "value": repr(value)}
    if isinstance(value, Program):
        if db is not None:
            db._ensure_program(value)
            return {"type": "program", "id": value.id}
        return {"type": "program_value", "value": encode(value.to_dict())}
    if isinstance(value, random.Random):
        return {"type": "rng", "value": encode(value.getstate())}
    if isinstance(value, deque):
        return {"type": "deque", "maxlen": value.maxlen, "value": [encode(v, db) for v in value]}
    if isinstance(value, (tuple, set, frozenset)):
        return {"type": type(value).__name__, "value": [encode(v, db) for v in value]}
    if isinstance(value, list):
        return [encode(v, db) for v in value]
    if isinstance(value, dict):
        # Encode keys too: metric caches and genealogies can have integer/tuple keys.
        factory = (
            "list" if isinstance(value, defaultdict) and value.default_factory is list else None
        )
        return {
            "type": "dict",
            "factory": factory,
            "value": [[encode(k, db), encode(v, db)] for k, v in value.items()],
        }
    allowed = _types()
    if type(value) in allowed.values():
        return {"type": "object", "class": type(value).__name__, "value": encode(vars(value), db)}
    if hasattr(value, "item"):
        return encode(value.item(), db)
    raise TypeError(
        f"Unsupported durable state: {type(value).__name__}. Use JSON-compatible values."
    )


def decode(value, db=None):
    if isinstance(value, list):
        return [decode(v, db) for v in value]
    if not isinstance(value, dict):
        return value
    kind = value["type"]
    data = value.get("value")
    if kind == "float":
        return float(data)
    if kind == "program":
        program = db._get_any(value["id"])
        if program is None:
            raise ValueError(f"Missing durable candidate {value['id']}")
        return program
    if kind == "program_value":
        return Program.from_dict(decode(data))
    if kind == "rng":
        rng = db.rng if db is not None else random.Random()
        rng.setstate(decode(data, db))
        return rng
    if kind == "dict":
        result = defaultdict(list) if value.get("factory") == "list" else {}
        result.update((decode(k, db), decode(v, db)) for k, v in data)
        return result
    if kind == "deque":
        return deque((decode(v, db) for v in data), maxlen=value["maxlen"])
    if kind in ("set", "frozenset", "tuple"):
        constructor = {"set": set, "frozenset": frozenset, "tuple": tuple}[kind]
        return constructor(decode(v, db) for v in data)
    if kind == "object":
        cls = _types()[value["class"]]
        result = cls.__new__(cls)
        result.__dict__.update(decode(data, db))
        return result
    raise ValueError(f"Unknown durable state type: {kind}")


def json_data(value):
    """Make candidate data JSONB-safe while retaining non-finite metrics and bytes."""
    import base64

    if isinstance(value, float) and not math.isfinite(value):
        return {"__skydiscover_float__": repr(value)}
    if isinstance(value, bytes):
        return {"__skydiscover_bytes__": base64.b64encode(value).decode("ascii")}
    if isinstance(value, tuple):
        return {"__skydiscover_tuple__": [json_data(v) for v in value]}
    if isinstance(value, dict):
        return {k: json_data(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_data(v) for v in value]
    if hasattr(value, "tolist"):
        return json_data(value.tolist())
    if hasattr(value, "item"):
        return json_data(value.item())
    return value


def restore_data(value):
    import base64

    if isinstance(value, list):
        return [restore_data(v) for v in value]
    if isinstance(value, dict):
        if set(value) == {"__skydiscover_float__"}:
            return float(value["__skydiscover_float__"])
        if set(value) == {"__skydiscover_bytes__"}:
            return base64.b64decode(value["__skydiscover_bytes__"])
        if set(value) == {"__skydiscover_tuple__"}:
            return tuple(restore_data(v) for v in value["__skydiscover_tuple__"])
        return {k: restore_data(v) for k, v in value.items()}
    return value
