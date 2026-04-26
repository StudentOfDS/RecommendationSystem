# RecommendationSystem

Production-leaning movie recommendation platform with Streamlit serving, SQLite storage, offline artifact training, and evaluation tooling.

## Architecture

- `app.py`: Streamlit UI for ingestion, recommendation serving, scraping, and evaluation tabs.
- `database.py`: SQLite schema + validated upserts for movies/ratings/reviews.
- `features.py`: TF-IDF + metadata + review-text feature construction.
- `recommender.py`: content-based, user CF, item CF (cosine/pearson), SVD, hybrid, and epsilon-greedy reranking.
- `train.py`: offline training pipeline that persists serving artifacts.
- `artifacts.py`: save/load helpers for feature + collaborative artifacts.
- `evaluation.py`: full benchmark runner and report writer.
- `scraper.py`: validated and policy-aware scraping pipeline with generic + site-specific parsers.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
streamlit run app.py
```

## Data schema

### Movies CSV
Required: `item_id`

Optional: `title,genres,description,tags,year`

### Ratings CSV
Required: `user_id,item_id,rating`

Optional: `timestamp`

Rating range is validated against configured scale `[0.5, 5.0]`.

## Offline training artifacts

Run:

```bash
python train.py --output-dir artifacts
```

Outputs:

- `artifacts/feature_artifacts.joblib`
- `artifacts/cf_artifacts.joblib`
- `artifacts/metadata.json`

These are loaded in app serving mode to avoid recomputation.

## Evaluation

- In-app: **Offline Evaluation** expander can run full benchmark.
- Programmatic report saved to `artifacts/evaluation_report.json`.
- Includes average Precision@K, Recall@K, NDCG@K across users and SVD RMSE/MAE.

## Scraping notes

- URL validation and `robots.txt` checks are enforced.
- Generic parser + basic site-specific handlers (IMDb/RottenTomatoes style selectors).
- Dynamic-page rendering (Playwright/Selenium), deeper per-site policy engines, and rich structured extraction are future roadmap items.

## Bandit notes

- Current reranker uses epsilon-greedy exploration with impression and reward mean tracking.
- Contextual models (LinUCB/Thompson) are not yet implemented and remain future work.

## Quality gates

- Tests: `pytest`
- Lint: `ruff`
- Type checks: `mypy`
- CI: `.github/workflows/ci.yml`
