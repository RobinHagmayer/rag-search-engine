import json
from pathlib import Path

DEFAULT_SEARCH_LIMIT = 5

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "movies.json"


def load_movies() -> list[dict]:
    with Path(DATA_PATH).open() as file:
        data = json.load(file)
    return data["movies"]
