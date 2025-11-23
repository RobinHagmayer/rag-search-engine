import pickle
import string
import sys
from collections import defaultdict

from nltk.stem import PorterStemmer

from .search_utils import CACHE_DIR, DEFAULT_SEARCH_LIMIT, load_movies, load_stopwords


class InvertedIndexLoadError(Exception):
    pass


class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.index_path = CACHE_DIR / "index.pkl"
        self.docmap_path = CACHE_DIR / "docmap.pkl"

    def build(self) -> None:
        movie_list = load_movies()
        for movie in movie_list:
            doc_id = movie["id"]
            doc_text = f"{movie['title']} {movie['description']}"
            self.docmap[doc_id] = movie
            self.__add_document(doc_id, doc_text)

    def save(self) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        with self.index_path.open("wb") as f:
            pickle.dump(self.index, f)

        with self.docmap_path.open("wb") as f:
            pickle.dump(self.docmap, f)

    def get_documents(self, term: str) -> list[int]:
        return sorted(self.index.get(term, set()))

    def __add_document(self, doc_id: int, text: str) -> None:
        token_list = tokenize_text(text)
        for token in token_list:
            self.index[token].add(doc_id)

    def load(self) -> None:
        try:
            with self.index_path.open("rb") as f:
                self.index = pickle.load(f)  # noqa: S301
        except FileNotFoundError as e:
            msg = f"Index file not found: {self.index_path}"
            raise InvertedIndexLoadError(msg) from e

        try:
            with self.docmap_path.open("rb") as f:
                self.docmap = pickle.load(f)  # noqa: S301
        except FileNotFoundError as e:
            msg = f"Docmap file not found: {self.docmap_path}"
            raise InvertedIndexLoadError(msg) from e


def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = InvertedIndex()

    try:
        idx.load()
    except InvertedIndexLoadError as e:
        sys.exit(f"Error loading the index: {e}")

    query_token_list = tokenize_text(query)
    results = []
    done = False
    for token in query_token_list:
        doc_id_list = idx.get_documents(token)

        for doc_id in doc_id_list:
            if len(results) == limit:
                done = True
                break
            results.append(idx.docmap.get(doc_id))

        if done:
            break

    return results


def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text  # noqa: RET504


def tokenize_text(text: str) -> list[str]:
    # Preparation
    text = preprocess_text(text)
    tokens = text.split()

    # Load resources
    stopwords = load_stopwords()
    stemmer = PorterStemmer()

    # Processing
    processed_tokens = []
    for token in tokens:
        # Filter empty tokens
        if token in stopwords:
            continue

        stemmed = stemmer.stem(token)
        processed_tokens.append(stemmed)

    return processed_tokens
