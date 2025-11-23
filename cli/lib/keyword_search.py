import string

from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []

    for movie in movies:
        pre_query = preprocess_text(query)
        pre_title = preprocess_text(movie["title"])

        if pre_query in pre_title:
            results.append(movie)
            if len(results) >= limit:
                break

    return results


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text
