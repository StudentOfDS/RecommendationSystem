from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ARTIFACT_DIR = BASE_DIR / "artifacts"
BANDIT_STATE_DIR = BASE_DIR / "bandit_state"
DB_PATH = DATA_DIR / "recommender.db"

# Data thresholds
MIN_USER_INTERACTIONS = 2
MIN_ITEM_INTERACTIONS = 2
RATING_SCALE_MIN = 0.5
RATING_SCALE_MAX = 5.0

# Feature engineering
TEXT_FIELDS = ["title", "description", "tags", "genres"]
MAX_TFIDF_FEATURES = 5000

# Recommenders
TOP_N_DEFAULT = 10
K_NEIGHBORS_DEFAULT = 20
SIMILARITY_METRIC = "cosine"
ALPHA_HYBRID_DEFAULT = 0.6
COLLAB_WEIGHTS = {
    "user_cf": 0.35,
    "item_cf": 0.35,
    "svd": 0.30,
}

# SVD
SVD_COMPONENTS = 40
SVD_RANDOM_STATE = 42

# Bandit
EPSILON = 0.15
BANDIT_REWARD_MIN = 0.0
BANDIT_REWARD_MAX = 1.0

# Scraper
SCRAPER_TIMEOUT_SECONDS = 10
SCRAPER_MAX_RETRIES = 2
SCRAPER_USER_AGENT = "RecommendationSystemBot/1.0 (respectful; educational)"

# Ensure dirs exist
for d in [DATA_DIR, ARTIFACT_DIR, BANDIT_STATE_DIR]:
    d.mkdir(parents=True, exist_ok=True)
