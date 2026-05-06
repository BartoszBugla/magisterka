from enum import Enum
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


FILE_LIMIT = 200 * 1024 * 1024  # 200MB

NON_ASPECT_COLUMNS: frozenset[str] = frozenset(
    {
        "text",
        "gmap_id",
        "lat",
        "lon",
        "lng",
        "latitude",
        "longitude",
        "time",
        "rating",
        "name",
        "category",
        "language",
        "word_count",
    }
)

TRAIN_ASPECTS = [
    "safety",
    "cleanliness",
    "infrastructure",
    "nature",
    "attractions",
    "heritage",
    "costs",
    "other",
]


class SentimentLabel(str, Enum):
    """ABSA label set — keep aligned with training and dataset validation."""

    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    NOTMENTIONED = "notmentioned"


SENTIMENT_LABELS = [e.value for e in SentimentLabel]

SENTIMENT_3 = [
    SentimentLabel.POSITIVE.value,
    SentimentLabel.NEUTRAL.value,
    SentimentLabel.NEGATIVE.value,
]
MENTION_LABELS = [SentimentLabel.NOTMENTIONED.value, "mentioned"]

MAX_LENGTH = 128


class ModelType(Enum):
    FINE_TUNED_BERT = "fine_tuned_bert"
    FINE_TUNED_DISTILBERT = "fine_tuned_distilbert"
    TFIDF_LSA = "tfidf_lsa"
    FINE_TUNED_DISTILBERT_SST = "fine_tuned_distilbert_sst"
    TFIDF_LSA_RF = "tfidf_lsa_rf"
    TEST_BERT_BASE_UNCASED_ABSA = "test_bert_base_uncased_absa"


RESULTS_REPOSITORY_DIR = "statics/results_repository"
MODEL_DIR = "statics/models"
