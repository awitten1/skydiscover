"""Short database operations; never hold a transaction across an LLM/evaluator await."""

from contextlib import nullcontext
from functools import wraps


def database_operation(method):
    if getattr(method, "_database_operation", False):
        return method

    @wraps(method)
    def wrapped(self, *args, **kwargs):
        operation = getattr(self, "operation", None)
        with operation() if operation else nullcontext():
            return method(self, *args, **kwargs)

    wrapped._database_operation = True
    return wrapped
