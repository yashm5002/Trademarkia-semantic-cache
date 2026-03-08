# Semantic Clustering & Caching API



A lightweight, cluster-aware semantic search engine and caching layer built from first principles. This API takes natural language queries, maps them into a 384-dimensional vector space, routes them using fuzzy probability distributions, and implements a bounded semantic cache—all without relying on heavy external vector databases or Redis.

## 🚀 Architecture & Tech Stack

- **Framework:** FastAPI (Python)
- **Embedding Model:** `BAAI/bge-small-en-v1.5` (via `sentence-transformers`)
- **Clustering Engine:** Gaussian Mixture Models (GMM) via `scikit-learn`
- **Cache Layer:** Native Python `OrderedDict` (O(1) LRU eviction policy)
- **Deployment:** Docker & Docker Compose

## 🧠 Design Justifications

1. **Embedding Selection:** `BGE-Small` was chosen for its state-of-the-art semantic retrieval accuracy while remaining compact (~133MB) and highly performant for CPU-bound API environments.
2. **Fuzzy Soft-Clustering (GMM):** Instead of K-Means hard labels, GMM provides a probability distribution. Text is nuanced; a query might belong to multiple topics. GMM allows us to identify a "dominant cluster" while respecting semantic overlap.
3. **Cluster-Aware Bounded Cache:** Standard caching requires comparing a new query against every cached item $O(N)$. By grouping queries into 15 macro-clusters, the cache search space is strictly bounded, allowing lightning-fast NumPy cosine similarity operations without needing FAISS or Milvus.
4. **Tunable Precision/Recall:** The cache threshold (e.g., `0.65`) explicitly dictates system behavior, allowing developers to prioritize aggressive caching (high recall/speed) or strict exact-matches (high precision).

## 🐳 Running with Docker

The application is fully containerized. To build and start the server:

```bash
# Build and run the container
docker-compose up --build
