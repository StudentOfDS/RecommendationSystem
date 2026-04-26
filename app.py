import pandas as pd
import streamlit as st

from artifacts import load_cf_artifacts, load_metadata
from config import ALPHA_HYBRID_DEFAULT, ARTIFACT_DIR, K_NEIGHBORS_DEFAULT, TOP_N_DEFAULT
from database import (
    add_manual_review,
    add_scraped_reviews,
    init_db,
    load_core_data,
    read_table,
    upsert_movies,
    upsert_ratings,
)
from evaluation import (
    evaluate_ranking_for_user,
    evaluate_svd_regression,
    run_offline_benchmark,
    save_evaluation_report,
    temporal_or_random_split,
)
from features import FeatureArtifacts, build_item_features, clean_ratings, load_feature_artifacts
from recommender import (
    EpsilonGreedyBandit,
    content_based_recommend,
    hybrid_recommend,
    item_based_cf_recommend,
    popularity_fallback,
    svd_recommend,
    user_based_cf_recommend,
)
from scraper import scrape_generic_reviews

st.set_page_config(page_title="Recommendation Platform", layout="wide")
st.title("🎬 End-to-End Movie Recommendation & Review Platform")

init_db()
bandit = EpsilonGreedyBandit()


@st.cache_data(show_spinner=False)
def build_feature_cache(movies_df: pd.DataFrame, reviews_df: pd.DataFrame) -> FeatureArtifacts:
    return build_item_features(movies_df, reviews=reviews_df)


def _load_data():
    movies_df, ratings_df, reviews_df = load_core_data()
    scraped_df = read_table("scraped_reviews")
    reviews_all = pd.concat(
        [
            reviews_df[["item_id", "review_text"]] if not reviews_df.empty else pd.DataFrame(columns=["item_id", "review_text"]),
            scraped_df[["item_id", "review_text"]] if not scraped_df.empty else pd.DataFrame(columns=["item_id", "review_text"]),
        ],
        ignore_index=True,
    )
    if not ratings_df.empty:
        ratings_df = clean_ratings(ratings_df, min_user_interactions=1, min_item_interactions=1)
    return movies_df, ratings_df, reviews_df, reviews_all


st.sidebar.header("Ingestion")
ratings_upload = st.sidebar.file_uploader("Upload ratings CSV (user_id,item_id,rating,timestamp optional)", type=["csv"])
items_upload = st.sidebar.file_uploader("Upload movies CSV (item_id,title,genres,description,tags,year)", type=["csv"])

if items_upload is not None:
    upsert_movies(pd.read_csv(items_upload))
    st.sidebar.success("Movies ingested")
if ratings_upload is not None:
    upsert_ratings(pd.read_csv(ratings_upload))
    st.sidebar.success("Ratings ingested")

movies_df, ratings_df, reviews_df, reviews_all = _load_data()

st.subheader("Data Snapshot")
c1, c2, c3 = st.columns(3)
c1.metric("Movies", len(movies_df))
c2.metric("Ratings", len(ratings_df))
c3.metric("Reviews", len(reviews_all))

if not movies_df.empty:
    st.dataframe(movies_df.head(20), use_container_width=True)

st.subheader("Manual Rating & Review")
with st.form("manual_form"):
    user_id = st.text_input("User ID", value="u_demo")
    item_id = st.text_input("Item ID")
    rating = st.slider("Rating", min_value=0.5, max_value=5.0, step=0.5, value=4.0)
    review_text = st.text_area("Optional review text")
    submit_manual = st.form_submit_button("Submit")
if submit_manual:
    if item_id:
        upsert_ratings(pd.DataFrame([{"user_id": user_id, "item_id": item_id, "rating": rating}]))
        if review_text.strip():
            add_manual_review(user_id, item_id, review_text.strip())
        reward = bandit.rating_to_reward(rating)
        bandit.update(user_id, item_id, reward)
        st.success("Feedback saved and bandit updated.")
    else:
        st.warning("Item ID is required.")

st.subheader("Optional Controlled Scraping")
with st.form("scrape_form"):
    scrape_item_id = st.text_input("Item ID for scraped reviews")
    scrape_url = st.text_input("Review page URL")
    scrape_click = st.form_submit_button("Scrape and save")
if scrape_click:
    if scrape_item_id and scrape_url:
        try:
            scraped = scrape_generic_reviews(scrape_item_id, scrape_url)
            add_scraped_reviews(scraped)
            st.success(f"Saved {len(scraped)} deduplicated scraped reviews")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Scraping failed: {exc}")
    else:
        st.warning("Both item_id and URL are required")

st.subheader("Recommendations")
if ratings_df.empty or movies_df.empty:
    st.info("Upload or add enough movie and rating data to generate recommendations.")
else:
    use_persisted = st.checkbox("Use persisted artifacts from artifacts/ (if available)", value=True)
    feature_artifacts = None
    cf_artifacts = None
    if use_persisted and (ARTIFACT_DIR / "feature_artifacts.joblib").exists() and (ARTIFACT_DIR / "cf_artifacts.joblib").exists():
        feature_artifacts = load_feature_artifacts(ARTIFACT_DIR / "feature_artifacts.joblib")
        cf_artifacts = load_cf_artifacts(ARTIFACT_DIR / "cf_artifacts.joblib")
        if (ARTIFACT_DIR / "metadata.json").exists():
            st.caption(f"Loaded artifacts metadata: {load_metadata(ARTIFACT_DIR / 'metadata.json')}")
    else:
        feature_artifacts = build_feature_cache(movies_df, reviews_all)

    users = sorted(ratings_df["user_id"].astype(str).unique().tolist())
    rec_user = st.selectbox("Select user", options=users)

    algo = st.selectbox("Algorithm", options=["hybrid", "content", "user_cf", "item_cf", "svd", "popularity"], index=0)
    top_n = st.slider("Top N", min_value=3, max_value=30, value=TOP_N_DEFAULT)
    alpha = st.slider("Hybrid alpha (collab weight)", min_value=0.0, max_value=1.0, value=float(ALPHA_HYBRID_DEFAULT))
    k_neighbors = st.slider("K neighbors", min_value=5, max_value=80, value=K_NEIGHBORS_DEFAULT)

    if st.button("Generate recommendations"):
        if algo == "hybrid":
            recs = hybrid_recommend(rec_user, ratings_df, feature_artifacts, top_n=top_n, alpha=alpha, cf_artifacts=cf_artifacts)
        elif algo == "content":
            recs = content_based_recommend(rec_user, ratings_df, feature_artifacts, top_n=top_n)
        elif algo == "user_cf":
            recs = user_based_cf_recommend(rec_user, ratings_df, top_n=top_n, k_neighbors=k_neighbors, cf_artifacts=cf_artifacts)
        elif algo == "item_cf":
            recs = item_based_cf_recommend(rec_user, ratings_df, top_n=top_n, k_neighbors=k_neighbors, cf_artifacts=cf_artifacts)
        elif algo == "svd":
            recs = svd_recommend(rec_user, ratings_df, top_n=top_n, cf_artifacts=cf_artifacts)
        else:
            recs = popularity_fallback(ratings_df, top_n=top_n)

        recs = bandit.rerank(rec_user, recs)
        out = pd.DataFrame([{"item_id": r.item_id, "score": r.score, "reason": r.reason} for r in recs])
        st.dataframe(out, use_container_width=True)

        train_df, test_df = temporal_or_random_split(ratings_df, test_size=0.2)
        test_user = test_df[test_df["user_id"].astype(str) == str(rec_user)]
        if not test_user.empty and not out.empty:
            metrics = evaluate_ranking_for_user(out["item_id"].astype(str).tolist(), test_user, k=min(10, len(out)))
            st.write("Evaluation (demo split)", metrics)

with st.expander("Offline Evaluation"):
    if st.button("Run full benchmark") and not ratings_df.empty and not movies_df.empty:
        f_art = build_feature_cache(movies_df, reviews_all)
        tr, te = temporal_or_random_split(ratings_df, test_size=0.2)
        bench_df = run_offline_benchmark(tr, te, f_art, k=10)
        svd_metrics = evaluate_svd_regression(tr, te)
        report = {"ranking": bench_df.to_dict(orient="records"), "svd": svd_metrics}
        save_evaluation_report(report)
        st.dataframe(bench_df, use_container_width=True)
        st.write("SVD RMSE/MAE", svd_metrics)
        st.success("Saved report to artifacts/evaluation_report.json")
