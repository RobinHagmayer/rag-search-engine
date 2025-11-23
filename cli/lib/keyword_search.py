import math
import pickle
import string
import sys
from collections import Counter, defaultdict

from nltk.stem import PorterStemmer

from .search_utils import (
    BM25_B,
    BM25_K1,
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    format_search_result,
    load_movies,
    load_stopwords,
)


class InvertedIndexLoadError(Exception):
    pass


class InvertedIndexTermFrequenciesError(Exception):
    pass


class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.term_frequencies = defaultdict(Counter)
        self.doc_lengths = {}
        self.index_path = CACHE_DIR / "index.pkl"
        self.docmap_path = CACHE_DIR / "docmap.pkl"
        self.tf_path = CACHE_DIR / "term_frequencies.pkl"
        self.doc_lengths_path = CACHE_DIR / "doc_lengths.pkl"

    def build(self) -> None:
        movie_list = load_movies()
        for movie in movie_list:
            doc_id = movie["id"]
            doc_text = f"{movie['title']} {movie['description']}"
            self.docmap[doc_id] = movie
            self.__add_document(doc_id, doc_text)

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

        try:
            with self.tf_path.open("rb") as f:
                self.term_frequencies = pickle.load(f)  # noqa: S301
        except FileNotFoundError as e:
            msg = f"Term frequencies file not found: {self.tf_path}"
            raise InvertedIndexLoadError(msg) from e

        try:
            with self.doc_lengths_path.open("rb") as f:
                self.doc_lengths = pickle.load(f)  # noqa: S301
        except FileNotFoundError as e:
            msg = f"Document lengths file not found: {self.doc_lengths_path}"
            raise InvertedIndexLoadError(msg) from e

    def save(self) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        with self.index_path.open("wb") as f:
            pickle.dump(self.index, f)

        with self.docmap_path.open("wb") as f:
            pickle.dump(self.docmap, f)

        with self.tf_path.open("wb") as f:
            pickle.dump(self.term_frequencies, f)

        with self.doc_lengths_path.open("wb") as f:
            pickle.dump(self.doc_lengths, f)

    def get_bm25_idf(self, term: str) -> float:
        term_list = tokenize_text(term)
        if len(term_list) != 1:
            raise ValueError

        term = term_list[0]
        doc_count = len(self.docmap)
        df = len(self.index[term])

        return math.log((doc_count - df + 0.5) / (df + 0.5) + 1)

    def get_bm25_tf(self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
        doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self.__get_avg_doc_length()

        length_norm = 1 - b + b * (doc_length / avg_doc_length) if avg_doc_length > 0 else 1

        try:
            tf = self.get_tf(doc_id, term)
        except InvertedIndexTermFrequenciesError:
            return 0.0

        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def bm25(self, doc_id: int, term: str) -> float:
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)
        return bm25_tf * bm25_idf

    def bm25_search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
        token_list = tokenize_text(query)

        scores: dict[int, float] = {}
        for doc_id in self.docmap:
            score = 0.0
            for token in token_list:
                bm25_score = self.bm25(doc_id, token)
                score += bm25_score
            scores[doc_id] = score

        sorted_scores: list[tuple[int, float]] = sorted(scores.items(), key=lambda item: item[1], reverse=True)

        results = []
        for doc_id, score in sorted_scores[:limit]:
            doc = self.docmap[doc_id]
            formatted_result = format_search_result(
                doc_id=doc["id"],
                title=doc["title"],
                document=doc["description"],
                score=score,
            )
            results.append(formatted_result)

        return results

    def get_documents(self, term: str) -> list[int]:
        return sorted(self.index.get(term, set()))

    def get_idf(self, term: str) -> float:
        term_list = tokenize_text(term)
        if len(term_list) != 1:
            raise ValueError

        term = term_list[0]
        doc_count = len(self.docmap.keys())
        term_doc_count = len(self.index[term])

        return math.log((doc_count + 1) / (term_doc_count + 1))

    def get_tf(self, doc_id: int, term: str) -> int:
        token_list = tokenize_text(term)
        if len(token_list) != 1:
            msg = "Can only get term frequency for one token"
            raise InvertedIndexTermFrequenciesError(msg)
        token = token_list[0]

        doc_id_list = self.index.get(token)
        if doc_id not in doc_id_list:  # ty: ignore[unsupported-operator]
            msg = "Term does not exist in given document id"
            raise InvertedIndexTermFrequenciesError(msg)

        return self.term_frequencies.get(doc_id).get(token)  # ty: ignore[invalid-return-type, possibly-missing-attribute]

    def get_tfidf(self, doc_id: int, term: str) -> float:
        token_list = tokenize_text(term)
        if len(token_list) != 1:
            msg = "Can only get term frequency for one token"
            raise InvertedIndexTermFrequenciesError(msg)
        token = token_list[0]

        tf = self.get_tf(doc_id, token)
        idf = self.get_idf(token)

        return tf * idf

    def __add_document(self, doc_id: int, text: str) -> None:
        token_list = tokenize_text(text)
        token_count = len(token_list)
        self.doc_lengths[doc_id] = token_count
        self.term_frequencies[doc_id].update(token_list)  # ty:ignore[possibly-missing-attribute]
        for token in token_list:
            self.index[token].add(doc_id)

    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths or len(self.doc_lengths) == 0:
            return 0.0

        document_count = len(self.doc_lengths)
        total_length = 0

        for length in self.doc_lengths.values():
            total_length += length

        return total_length / document_count


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


def tf_command(doc_id: int, term: str) -> int:
    idx = InvertedIndex()

    try:
        idx.load()
    except InvertedIndexLoadError as e:
        sys.exit(f"Error loading the index: {e}")

    try:
        tf = idx.get_tf(doc_id, term)
    except InvertedIndexTermFrequenciesError as e:
        print(f"Error getting term frequencies: {e}")  # noqa: T201
        return 0

    return tf


def idf_command(term: str) -> float:
    idx = InvertedIndex()

    try:
        idx.load()
    except InvertedIndexLoadError as e:
        sys.exit(f"Error loading the index: {e}")

    return idx.get_idf(term)


def tfidf_command(doc_id: int, term: str) -> float:
    idx = InvertedIndex()

    try:
        idx.load()
    except InvertedIndexLoadError as e:
        sys.exit(f"Error loading the index: {e}")

    return idx.get_tfidf(doc_id, term)


def bm25_idf_command(term: str) -> float:
    idx = InvertedIndex()

    try:
        idx.load()
    except InvertedIndexLoadError as e:
        sys.exit(f"Error loading the index: {e}")

    return idx.get_bm25_idf(term)


def bm25_tf_command(doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
    idx = InvertedIndex()

    try:
        idx.load()
    except InvertedIndexLoadError as e:
        sys.exit(f"Error loading the index: {e}")

    return idx.get_bm25_tf(doc_id, term, k1, b)


def bm25_search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = InvertedIndex()

    try:
        idx.load()
    except InvertedIndexLoadError as e:
        sys.exit(f"Error loading the index: {e}")

    return idx.bm25_search(query, limit)


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
