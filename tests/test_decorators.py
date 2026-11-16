"""Verify exception reporting and enforce the repository's decorator convention."""

from inspect import signature
from pathlib import Path
from src.utils.decorators import handle_exceptions

import ast
import pytest


@handle_exceptions
def test_handler_preserves_signature_and_reraises(capsys):
    error = ValueError("example failure")

    @handle_exceptions
    def fail(value: int) -> None:
        """Example function with an inspectable signature."""
        raise error

    assert fail.__name__ == "fail"
    assert fail.__doc__ == "Example function with an inspectable signature."
    assert signature(fail) == signature(fail.__wrapped__)
    with pytest.raises(ValueError) as caught:
        fail(1)
    assert caught.value is error
    report = capsys.readouterr().out
    assert "Exception caught in function: fail" in report
    assert "Error: example failure" in report


@handle_exceptions
def test_all_functions_use_exception_handler():
    root = Path(__file__).resolve().parents[1]
    paths = list(root.glob("*.py"))
    for folder in ("src", "tests"):
        paths.extend((root / folder).rglob("*.py"))

    missing = []
    for path in paths:
        tree = ast.parse(path.read_text())
        bootstrap = set()
        if path == root / "src/utils/decorators.py":
            handler = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                           and node.name == "handle_exceptions")
            bootstrap.add(handler)
            bootstrap.update(node for node in handler.body if isinstance(node, ast.FunctionDef))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node in bootstrap:
                continue
            if not any(isinstance(item, ast.Name) and item.id == "handle_exceptions"
                       for item in node.decorator_list):
                missing.append(f"{path.relative_to(root)}:{node.lineno}: {node.name}")
    assert not missing, "Missing @handle_exceptions:\n" + "\n".join(missing)
