# 00 — Installation

This project uses [uv](https://docs.astral.sh/uv/) for package and environment management.
uv manages the Python version, the virtual environment (`.venv`) and the dependencies
with a single tool, replacing the `pip` + `venv` + `pyenv` combination.

## Why uv instead of pip?

| | pip + venv | uv |
|---|---|---|
| Speed | Slow | 10–100x faster |
| Installing Python | Needs a separate tool (pyenv etc.) | Built in (`uv python install`) |
| Virtual environment | Created and activated by hand | Automatic (`.venv`) |
| Lockfile | None (`requirements.txt` kept by hand) | `uv.lock`, everyone installs the same versions |

Reproducible results matter in ML projects, so library versions need to be pinned;
`uv.lock` guarantees that.

## 1. Install uv

### macOS

With Homebrew:

```bash
brew install uv
```

or with the official install script:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Windows (PowerShell)

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Open a new terminal afterwards and verify:

```bash
uv --version
```

## 2. Set up the project

Clone the repository, enter the project folder, then run:

```bash
uv sync
```

This single command:

1. Downloads the Python version from `.python-version` (3.12) if it is missing.
2. Creates the virtual environment in `.venv/`.
3. Installs the exact dependency versions from `uv.lock` (dev tools included).
4. Installs the `quant_ml` package in editable mode.

You do not need to install Python separately.

## 3. Run

There is no need to activate the virtual environment; `uv run` executes in the right one:

```bash
uv run python main.py
```

If it prints `quant_ml is ready`, the installation works.

To activate the environment the classic way instead:

```bash
source .venv/bin/activate
```

(Windows: `.venv\Scripts\activate`)

## 4. Managing dependencies

Add a library (`pyproject.toml` and `uv.lock` are updated automatically):

```bash
uv add xgboost
```

Add a tool that is only needed during development:

```bash
uv add --dev mypy
```

Remove a library:

```bash
uv remove xgboost
```

Upgrade the locked versions:

```bash
uv lock --upgrade
```

Do not use `pip install`; the environment would drift away from `uv.lock`.
Always commit `pyproject.toml` and `uv.lock` together.

## 5. Dev tools

Lint:

```bash
uv run ruff check .
```

Format:

```bash
uv run ruff format .
```

Tests (once a `tests/` folder exists):

```bash
uv run pytest
```

## 6. Jupyter notebooks

`ipykernel` is installed as a dev dependency. In VS Code, open a notebook and select
the Python inside `.venv` as the kernel. For JupyterLab in the browser:

```bash
uv run --with jupyterlab jupyter lab
```

## 7. Project structure

```
quant-ml/
├── main.py              # Entry point
├── pyproject.toml       # Project definition and dependencies
├── uv.lock              # Locked dependency versions (do not edit by hand)
├── .python-version      # Python version (3.12)
├── src/quant_ml/        # Source code (importable as `quant_ml`)
└── docs/                # Documentation
```

## Troubleshooting

- **`uv: command not found`** — Restart the terminal. If it is still missing, add
  `~/.local/bin` to your `PATH`.
- **`ModuleNotFoundError: quant_ml`** — You are running with the system Python instead
  of `uv run`. Use `uv run python main.py` or activate `.venv`.
- **Broken environment** — Delete the `.venv` folder and run `uv sync` again;
  `uv.lock` restores the same environment.
