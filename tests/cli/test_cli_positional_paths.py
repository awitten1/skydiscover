"""Tests for CLI positional path parsing."""

import sys

import pytest

from skydiscover.optimize.cli import parse_args


def test_parse_args_single_path_uses_evaluation_file(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["skydiscover optimize", "evaluate.py"])

    args = parse_args()

    assert args.initial_program is None
    assert args.evaluation_file == "evaluate.py"


def test_parse_args_two_paths_use_initial_and_evaluation(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["skydiscover optimize", "seed.py", "evaluate.py"])

    args = parse_args()

    assert args.initial_program == "seed.py"
    assert args.evaluation_file == "evaluate.py"


def test_parse_args_rejects_more_than_two_paths(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["skydiscover optimize", "a.py", "b.py", "c.py"])

    with pytest.raises(SystemExit) as exc_info:
        parse_args()

    assert exc_info.value.code == 2


def test_stored_task_accepts_evaluator_and_optional_seed():
    args = parse_args(["eval.py", "--task", "task-id", "--initial-program", "seed.py"])
    assert args.task == "task-id"
    assert args.evaluation_file == "eval.py"
    assert args.initial_program == "seed.py"
    assert args.config is None


def test_task_and_resume_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        parse_args(["--task", "task-id", "--resume", "run-id"])
