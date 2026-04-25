# RecommendationSystem

End-to-end movie recommendation platform with:

- Streamlit UI (`app.py`) for ingestion, rating, reviews, algorithm selection, recommendations, and feedback.
- SQLite persistence (`database.py`) with normalized tables for users, movies, ratings, manual reviews, and scraped reviews.
- Controlled scraping (`scraper.py`) with retries, timeout, deduplication, and normalization.
- Feature engineering (`features.py`) using TF-IDF + one-hot + numeric scaling.
- Recommendation engines (`recommender.py`):
  - Content-based filtering
  - User-based collaborative filtering
  - Item-based collaborative filtering
  - Matrix factorization (Truncated SVD)
  - Hybrid weighted recommender
  - Epsilon-greedy bandit reranker with incremental online learning
- Evaluation (`evaluation.py`) with Precision@K, Recall@K, NDCG@K, RMSE, and MAE helpers.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Suggested dataset

Use MovieLens ratings + movie metadata CSVs, then upload in the sidebar:

- ratings: `user_id,item_id,rating[,timestamp]`
- items: `item_id,title,genres,description,tags,year`

## Notes

- Collaborative filtering consumes **numeric ratings only**.
- Raw text reviews are stored separately and can be transformed into features.
- Bandit state is file-backed under `bandit_state/` for lightweight local persistence.
