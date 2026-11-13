import faiss
import numpy as np


def create_vector_store(embeddings):
    embeddings = np.ascontiguousarray(embeddings, dtype="float32")

    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("No embeddings to index.")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index


def search_vector_store(index, query_embedding, k=5):
    query_embedding = np.ascontiguousarray(query_embedding, dtype="float32")

    # Make sure the query is 2D: shape (1, dimension)
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    # Don't ask for more results than the index holds
    k = min(k, index.ntotal)

    distances, indices = index.search(query_embedding, k)
    return distances, indices