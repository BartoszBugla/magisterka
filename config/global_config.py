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

TARGET_CATEGORIES = [
    "Park",
    "City park",
    "Playground",
    "Sports complex",
    "Public swimming pool",
    "Recreation center",
    "Train station",
    "Transit station",
    "Bus station",
    "Subway station",
    "Museum",
    "Art gallery",
    "Historical landmark",
    "Monument",
    "Castle",
    "Town square",
    "Plaza",
    "Tourist attraction",
    "Historical place",
    "Shopping mall",
    "Flea market",
    "Farmers' market",
    "Bazaar",
]


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


class ModelType(Enum):
    FINE_TUNED_BERT = "fine_tuned_bert"
    FINE_TUNED_DISTILBERT = "fine_tuned_distilbert"


RESULTS_REPOSITORY_DIR = "statics/results_repository"
MODEL_DIR = "statics/models"
