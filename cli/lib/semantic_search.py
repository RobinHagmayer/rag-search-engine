import numpy as np
from sentence_transformers import SentenceTransformer

from .search_utils import CACHE_DIR, load_movies

# ruff: noqa: T201


class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.documents_map = {}

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

    def generate_embedding(self, text: str) -> str:
        if not text or not text.strip():
            msg = "'text' must be a valid non-empty string"
            raise ValueError(msg)

        embedding = self.model.encode([text])

        return embedding[0]


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


def embed_text(text: str) -> None:
    search_instance = SemanticSearch()

    embedding = search_instance.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")
