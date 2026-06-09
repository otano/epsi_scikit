# epsi-scikit

ML/data-science playground (EPSI school project). Simple scikit-learn pipeline on French road accident data.

## Commands

- **Sync deps**: `uv sync` (uses `uv` — not `pip` or `pipenv`)
- **Run pipeline**: `uv run python main.py`
- **Run API**: `uv run python app.py`
- **Add dep**: `uv add <package>`

## Project structure

| Path | Purpose |
|------|---------|
| `main.py` | ML pipeline — `uv run python main.py` (saves `model.joblib`, `scaler.joblib`, `imputer.joblib`) |
| `app.py` | Flask-RESTX API server (port 9090) — `uv run python app.py` |
| `data/donnees_agregees_de_2005_a_2010.csv` | ~460K rows of French road accident data (target column `grav`) |
| `data/descriptif_des_variables_pour_le_fichier_de_2005_a_2010.pdf` | Variable descriptions |

## API endpoints

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/api/health` | GET | Health check |
| `/api/dbhealth` | GET | PostgreSQL connection check |
| `/predict/` | POST | Régression — prédit l'indice `grav` (R² ≈ 0.01) |
| `/predict-severity/` | POST | Classification — `léger` / `grave` / `mortel` (acc. 65%) |
| `/cluster/` | POST | Clustering — typologie d'accident (2 profils) |

Tous les endpoints POST attendent 11 features numériques : `org`, `dep`, `com`, `lat`, `long`, `catr`, `voie`, `v1`, `pr1`, `typenumero`, `distancemetre`.

`/cluster/` n'utilise que 4 features : `lat`, `long`, `catr`, `distancemetre`.

## PostgreSQL (Docker)

- Start: `docker compose up -d` (PostgreSQL 16 on port 5432, db `epsi_scikit`, user/pass `epsi`/`epsi`)
- Stop: `docker compose down`
- Wipe data: `docker compose down -v`
- Env vars to override: `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASS`

## Notes

- `main.py` trains 3 models: RandomForestRegressor, RandomForestClassifier (séverité), KMeans (clustering). Use `uv run python main.py` to retrain.
- No tests, no CI, no linting/formatting/typecheck config. `pyproject.toml` has only project metadata and deps.
- Repo has zero commits — no history to search.
- `.venv/` is gitignored. Activate with `source .venv/bin/activate` or use `uv run`.
- Python 3.13 required (see `.python-version`).
