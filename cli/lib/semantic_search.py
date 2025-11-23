import numpy as np
from sentence_transformers import SentenceTransformer

from .search_utils import CACHE_DIR, load_movies

# ruff: noqa: T201


class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents: list[dict] = None
        self.documents_map = {}

    def search(self, query: str, limit: int) -> list[dict]:
        if self.embeddings is None or self.embeddings.size == 0:
            msg = "No embeddings loaded. Call `load_or_create_embeddings` first."
            raise ValueError(msg)

        if self.documents is None or len(self.documents) == 0:
            msg = "No documents loaded. Call `load_or_create_embeddings` first."
            raise ValueError(msg)

        query_embedding = self.generate_embedding(query)

        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            score = cosine_similarity(query_embedding, doc_embedding)
            similarities.append((score, self.documents[i]))

        similarities.sort(key=lambda item: item[0], reverse=True)

        results = []
        for score, doc in similarities[:limit]:
            results.append(
                {
                    "score": score,
                    "title": doc["title"],
                    "description": doc["description"],
                }
            )

        return results

    def load_or_create_embeddings(self, documents: list[dict]):  # noqa: ANN201
        self.documents = documents

        for doc in documents:
            self.documents_map[doc["id"]] = doc

        file_path = CACHE_DIR / "movie_embeddings.npy"
        if file_path.is_file():
            self.embeddings = np.load(file_path)
            if len(self.embeddings) == len(documents):
                return self.embeddings

        return self.build_embeddings(documents)

    def build_embeddings(self, documents: list[dict]):  # noqa: ANN201
        self.documents = documents

        doc_as_string_list = []
        for doc in documents:
            self.documents_map[doc["id"]] = doc
            doc_as_string = f"{doc['title']}: {doc['description']}"
            doc_as_string_list.append(doc_as_string)

        self.embeddings = self.model.encode(doc_as_string_list, show_progress_bar=True)
        file_path = CACHE_DIR / "movie_embeddings.npy"
        np.save(file_path, self.embeddings)

        return self.embeddings

    def generate_embedding(self, text: str):  # noqa: ANN201
        if not text or not text.strip():
            msg = "'text' must be a valid non-empty string"
            raise ValueError(msg)

        embedding = self.model.encode([text])

        return embedding[0]


def semantic_search(query: str, limit: int = 5) -> None:
    search_instance = SemanticSearch()

    movies = load_movies()
    search_instance.load_or_create_embeddings(movies)

    result_list = search_instance.search(query, limit)

    print(f"Query: {query}")
    print(f"Top {len(result_list)} results:")
    print()

    for i, result in enumerate(result_list, 1):
        print(f"{i}. {result['title']} (score: {result['score']:.4f})")
        print(f"   {result['description'][:100]}...")
        print()


def verify_model() -> None:
    search_instance = SemanticSearch()

    print(f"Model loaded: {search_instance.model}")
    print(f"Max sequence length: {search_instance.model.max_seq_length}")


def verify_embeddings() -> None:
    search_instance = SemanticSearch()

    movies = load_movies()
    embeddings = search_instance.load_or_create_embeddings(movies)

    print(f"Number of docs:   {len(movies)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")


def embed_query_text(query: str) -> None:
    search_instance = SemanticSearch()

    embedding = search_instance.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def embed_text(text: str) -> None:
    search_instance = SemanticSearch()

    embedding = search_instance.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")
