# Dauphin SFN

## Abstract

<!-- Write the abstract here. -->

## Project details

<!-- Add the research objective, methods, results, and references here. -->

## Setup

Use Python **3.12**. The repository records **3.12.10** in `.python-version`.
Run the following commands from the repository root unless noted otherwise.

### 1. Clone the repository

```bash
git clone https://github.com/KengoKato02/dauphin-sfn.git
cd dauphin-sfn
```

If you already have a local checkout, update it with `git pull`.

### 2. Create and activate the virtual environment

On macOS or Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

On Windows, using PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Activate this environment again whenever you open a new terminal for the project.

### 3. Install the project and notebook dependencies

With `.venv` active:

```bash
python -m pip install -e ".[dev]"
```

This uses the dependencies in `pyproject.toml`. The `dev` extra includes
IPyKernel, JupyterLab, and pytest. The editable install (`-e`) lets Python use
your source changes without reinstalling the project.

For runtime dependencies only, without the notebook tools, use
`python -m pip install -e .` instead.

All project modules are installed under the `src` package. For example, use
`from src.utils.decorators import handle_exceptions`, not `from utils.decorators`.
The editable install makes these imports available outside the repository folder too.
After changing package configuration or moving packages, rerun the install command
and restart the notebook kernel.

### 4. Register the notebook kernel

Run this from the activated `.venv`:

```bash
python -m ipykernel install --sys-prefix --name dauphin-sfn --display-name "Python (dauphin-sfn)"
```

This registers a kernel using the Python interpreter and packages inside `.venv`.
The kernel registration is also stored inside that environment.

### 5. Open and run the notebook

In VS Code:

1. Open the repository folder.
2. Install Microsoft's [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
   and [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) extensions.
3. Open [the experiment notebook](notebooks/01_figure3_msgd_newton.ipynb).
4. Click **Select Kernel** in the notebook's upper-right corner.
5. Select **Python (dauphin-sfn)**, or choose the Python environment inside this
   repository's `.venv` folder.
6. Run the cells in order, or click **Run All**.

The notebook kernel is selected separately from the terminal environment.
Installing the project in one environment does not install it in a different kernel.
To confirm the notebook uses `.venv`, run this in a cell:

```python
import sys
print(sys.executable)
```

The printed path should point inside this repository's `.venv` directory.
See the [VS Code notebook documentation](https://code.visualstudio.com/docs/datascience/jupyter-notebooks)
for kernel selection.

Alternatively, launch JupyterLab from the activated environment:

```bash
jupyter lab
```

Open the notebook under `notebooks/` and select **Python (dauphin-sfn)**.

### First-run data preparation

The notebook downloads and prepares a dataset if its processed training file
is missing. The first CIFAR-10 run can therefore take longer than later runs.
A small training limit does not reduce the initial download or preparation.

Downloaded and processed data are stored under `data/` at the repository root.
Notebook figures and experiment records are saved under `results/figure3_starter/`.
These generated folders and `.venv/` are excluded from Git.
