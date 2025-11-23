import json
from pathlib import Path

DEFAULT_SEARCH_LIMIT = 5

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data"


def load_movies() -> list[dict]:
    with Path(DATA_PATH / "movies.json").open() as file:
        data = json.load(file)
    return data["movies"]


def load_stopwords() -> list[str]:
    with Path(DATA_PATH / "stopwords.txt").open() as file:
        return file.read().splitlines()
