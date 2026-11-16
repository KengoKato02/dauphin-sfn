"""Check that notebook imports work through the installed package."""

from pathlib import Path

import os
import ast
import sys
import subprocess


def test_notebook_imports_outside_repository(tmp_path):
    root = Path(__file__).resolve().parents[1]
    notebook = root / "notebooks" / "01_figure3_msgd_newton.ipynb"
    script = f"""
import json
from pathlib import Path

notebook = json.loads(Path({str(notebook)!r}).read_text())
namespace = {{}}
for cell in notebook['cells'][:3]:
    if cell['cell_type'] == 'code':
        exec(''.join(cell['source']), namespace)
assert namespace['ROOT'] == Path({str(root)!r})
assert namespace['DATA_DIR'] == Path({str(root / 'data')!r})
from src.pipelines.mnist import run_mnist_pipeline
from src.pipelines.cifar import run_cifar_pipeline
assert namespace['run_mnist_pipeline'] is run_mnist_pipeline
assert namespace['run_cifar_pipeline'] is run_cifar_pipeline
"""
    environment = os.environ.copy()
    environment["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    environment["MPLBACKEND"] = "Agg"
    # -I excludes the current directory and PYTHONPATH. The editable install
    # must supply src and all of its subpackages on its own.
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_project_imports_use_src_package():
    root = Path(__file__).resolve().parents[1]
    old_packages = {"utils", "pipelines", "algorithms", "models", "experiments", "viz",
                    "descent_algorithms"}
    invalid = []
    for path in (root / "src").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules = [node.module]
            elif isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            else:
                continue
            for module in modules:
                if module.split(".")[0] in old_packages:
                    invalid.append(f"{path.relative_to(root)}:{node.lineno}: {module}")
    assert not invalid, "Use src-qualified project imports:\n" + "\n".join(invalid)
