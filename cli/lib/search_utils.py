import json
from pathlib import Path

DEFAULT_SEARCH_LIMIT = 5

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache"


def load_movies() -> list[dict]:
    with Path(DATA_DIR / "movies.json").open() as file:
        data = json.load(file)
    return data["movies"]


def load_stopwords() -> list[str]:
    with Path(DATA_DIR / "stopwords.txt").open() as file:
        return file.read().splitlines()
