# epsi-scikit

ML/data-science playground (EPSI school project). Simple scikit-learn pipeline on French road accident data.

## Commands

- **Sync deps**: `uv sync` (uses `uv` — not `pip` or `pipenv`)
- **Run**: `uv run python main.py`
- **Add dep**: `uv add <package>`

## Project structure

| Path | Purpose |
|------|---------|
| `main.py` | Single entrypoint: load → preprocess → train_test_split → RandomForestRegressor → evaluate |
| `app.py` | Flask-RESTX API server (port 9090) — `uv run python app.py` |
| `data/donnees_agregees_de_2005_a_2010.csv` | ~460K rows of French road accident data (target column `grav`) |
| `data/descriptif_des_variables_pour_le_fichier_de_2005_a_2010.pdf` | Variable descriptions |

## Notes

- `main.py` has the pipeline defined but execution is commented out at the bottom — uncomment and set the target column name to run.
- No tests, no CI, no linting/formatting/typecheck config. `pyproject.toml` has only project metadata and deps.
- Repo has zero commits — no history to search.
- `.venv/` is gitignored. Activate with `source .venv/bin/activate` or use `uv run`.
- Python 3.13 required (see `.python-version`).
