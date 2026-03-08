<div align="center">

# 🧠 TradeMarkia
### Semantic Clustering & Caching Service

*A FastAPI microservice that turns the 20 Newsgroups corpus into a semantic clustering and in-memory caching engine — built from first principles.*

---

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-GMM-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

</div>

---

## 🔭 What is TradeMarkia?

TradeMarkia is a **FastAPI-based microservice** that turns the classic **20 Newsgroups** corpus into a **semantic clustering and in-memory caching engine**.

It combines:

- **Dense embeddings** via `BAAI/bge-small-en-v1.5`
- **Fuzzy clustering** via `GaussianMixture` (GMM) to model macro-topics
- A **cluster-aware, LRU-style semantic cache** implemented purely in Python

The result is an API that:

- Accepts a **natural-language query**
- Maps it into the learned cluster space
- Either serves a **cached semantic answer** (for similar past queries) or computes a new fuzzy distribution across clusters
- Exposes **cache performance metrics** and **flush controls**

> **Design philosophy:** This project is intentionally implemented **from first principles** (no Redis, no FAISS index for caching) to make the design decisions and trade-offs explicit.

---

## ⚡ How It Works

```
NL Query  ──►  BGE-Small Embed  ──►  GMM Cluster (15)  ──►  Cache Lookup  ──►  Fuzzy Distribution
                                                                  │
                                                          Hit ◄───┴───► Miss → Compute & Store
```

---

## 🏗️ High-Level Architecture

At a high level, TradeMarkia runs the following pipeline:

### 1 · Data Preparation `dataset.py`

- Loads the **20 Newsgroups** corpus from `Dataset/20_newsgroups.tar.gz`
- Aggressively **cleans 1990s Usenet noise**:
  - Strips headers (routing paths, institution names, timestamps)
  - Removes nested quotes and attribution lines
  - Removes signatures and email addresses
  - Normalizes whitespace
- Drops documents that become too short after cleaning (≤ 50 characters) to avoid noisy, "content-free" embeddings
- Retains the original 20-newsgroup labels only for later profiling/validation

### 2 · Semantic Embedding & Fuzzy Clustering `cluster.py`

- Uses `SentenceTransformer('BAAI/bge-small-en-v1.5')` as the embedding model
- Encodes each cleaned document into a dense vector
- Fits a **Gaussian Mixture Model (GMM)** with **15 clusters**:
  - 20 original categories → 15 macro-topics (merges semantically redundant newsgroups like similar hardware groups)
  - Provides **soft assignment** — each document (or query) has a **probability distribution across clusters**, not a single hard label

### 3 · Cluster-Aware Semantic Cache `cache.py`

- Pure Python implementation, using:
  - `dict` + `OrderedDict` for per-cluster buckets and LRU eviction
  - `numpy` for cosine similarity computations
- For each query:
  - Restricts search to **one cluster bucket** (O(K) rather than O(N) global)
  - Computes cosine similarity against cached vectors in that bucket
  - Uses a **tunable similarity threshold** to decide hits vs. misses
  - Maintains hits/misses and total entries for observability

### 4 · FastAPI Service `main.py`

- On startup:
  - Loads and cleans the dataset (`LocalDataLoader`)
  - Trains the `SemanticClusterer` on the clean corpus
  - Creates a `ClusterAwareCache` instance with a chosen similarity threshold
- Exposes HTTP endpoints:
  - `POST /query` – core semantic query + cache lookup
  - `GET /cache/stats` – cache performance metrics
  - `DELETE /cache` – flush cache + reset stats

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **Semantic Query Understanding** | BGE-Small embeddings capture intent rather than surface-level keyword overlaps |
| 🌀 **Fuzzy Topic Modeling** | GMM produces probability distributions over macro-topics, not just a single class — see how "torn" the model is between topics (e.g. religion vs. cosmology) |
| ⚡ **Cluster-Aware Semantic Cache** | Per-cluster buckets keep lookups bounded and efficient; uses cosine similarity with an explicit threshold and LRU eviction per cluster |
| 🌐 **API-First Design** | FastAPI service ready to be called from web apps, CLI tools, or other services — Swagger & ReDoc docs included |
| 🧠 **Fully In-Memory State** | No external cache (Redis, Memcached); embedding model, GMM, and cache all live inside the Python process |

---

## 📁 Project Structure

```
TradeMarkia/
│
├── main.py               # FastAPI app — wires dataset, clusterer, and cache together
├── dataset.py            # LocalDataLoader — reads & cleans the 20 Newsgroups tarball
├── cluster.py            # SemanticClusterer — embedding, GMM training, fuzzy distributions
├── cache.py              # ClusterAwareCache — pure-Python LRU semantic cache
├── view_dataset.py       # Utility to inspect raw Usenet documents from the tarball
│
├── requirements.txt      # Python runtime dependencies
├── Dockerfile            # Production-oriented container image
├── docker-compose.yaml   # Compose setup — mounts Dataset/ into the container
├── setup.bat             # Windows helper script to launch uvicorn
│
└── Dataset/
    ├── 20_newsgroups.tar.gz        # ← place dataset here
    ├── 20newsgroups.html           # Original dataset metadata (UCI KDD Archive)
    └── 20newsgroups.data.html      # Dataset description & licensing
```

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.10+ |
| **Web Framework** | FastAPI |
| **Embeddings** | `sentence-transformers` — `BAAI/bge-small-en-v1.5` |
| **Clustering** | `scikit-learn` — `GaussianMixture` |
| **Numerical Computing** | NumPy |
| **ASGI Server** | Uvicorn |
| **Infrastructure** | Docker & docker-compose |
| **Dataset** | 20 Newsgroups (UCI KDD Archive) |

---

## 🚀 Getting Started (Local Development)

### Prerequisites

- Python **3.10+**
- `git`
- (Optional) `virtualenv` or equivalent

### 1 · Clone the Repository

```bash
git clone <your-repo-url> TradeMarkia
cd TradeMarkia
```

### 2 · Create and Activate a Virtual Environment

**Windows (PowerShell / cmd):**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3 · Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install: `fastapi`, `uvicorn`, `sentence-transformers`, `scikit-learn`, `numpy`, `pandas`, `watchfiles`

### 4 · Download / Place the Dataset

The service expects: `Dataset/20_newsgroups.tar.gz`

If it is not already present:

```bash
mkdir -p Dataset
```

Download the **20 Newsgroups** tarball from the UCI KDD Archive (see the HTML files in `Dataset/` for links), and save it as `Dataset/20_newsgroups.tar.gz`.

### 5 · Run the API with Uvicorn

```bash
uvicorn main:app --reload
```

Or, on Windows, you can use the helper script:

```bash
setup.bat
```

The service will start (by default) on `http://127.0.0.1:8000`

📖 Interactive docs:
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

---

## 🐳 Running with Docker

### 1 · Build the Image

```bash
docker build -t trademarkia-service .
```

### 2 · Run via `docker-compose`

Ensure that the `Dataset` directory (containing `20_newsgroups.tar.gz`) exists on your host at `./Dataset`. Then run:

```bash
docker-compose up --build
```

This will:
- Build the image (if not already built)
- Start the FastAPI app at port **8000**
- Mount `./Dataset` into `/app/Dataset` inside the container

Access the API at: `http://localhost:8000/docs`

To stop:

```bash
docker-compose down
```

---

## 📡 API Reference

### `POST /query`

Process a natural-language query, perform fuzzy clustering, and consult the semantic cache.

**Request body:**
```json
{
  "query": "string"
}
```

**Example:**
```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"Did God create the universe, or was it the Big Bang?\"}"
```

**Cache miss response:**
```json
{
  "query": "Did God create the universe, or was it the Big Bang?",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": null,
  "result": [0.0012, 0.0345, "..."],
  "dominant_cluster": 7
}
```

**Cache hit response:**
```json
{
  "query": "Did God create the universe, or was it the Big Bang?",
  "cache_hit": true,
  "matched_query": "Is the Big Bang compatible with religious beliefs?",
  "similarity_score": 0.9321,
  "result": [0.0008, 0.0123, "..."],
  "dominant_cluster": 7
}
```

> - **`result`** — fuzzy distribution (length = number of clusters, default 15)
> - **`dominant_cluster`** — index of the cluster with the highest probability
> - **`cache_hit`**, **`matched_query`**, **`similarity_score`** — expose cache behavior

---

### `GET /cache/stats`

Retrieve high-level cache performance metrics.

```bash
curl "http://127.0.0.1:8000/cache/stats"
```

```json
{
  "total_entries": 42,
  "hit_count": 30,
  "miss_count": 12,
  "hit_rate": 0.714
}
```

> Stats are tracked in-process and reset when the service restarts or when the cache is flushed.

---

### `DELETE /cache`

Flushes all cache buckets and resets internal stats.

```bash
curl -X DELETE "http://127.0.0.1:8000/cache"
```

```json
{
  "message": "Cache flushed and stats reset."
}
```

---

## 🧩 Design Decisions & Rationale

The codebase is heavily annotated with reasoning comments. Key decisions:

<details>
<summary><b>🔡 Embedding Model — BGE-Small (<code>BAAI/bge-small-en-v1.5</code>)</b></summary>

Chosen as a compact (~133MB), high-quality sentence embedding model. Well-suited for CPU-bound environments and production APIs.

</details>

<details>
<summary><b>🌀 Fuzzy Clustering with GMM</b></summary>

`GaussianMixture` (with diagonal covariance) provides **soft cluster assignments**. Reflects the reality that many documents lie at the boundary between topics. Number of clusters (`n_components=15`) is **deliberately smaller** than the original 20 labels to merge redundant categories and discover macro-topics.

</details>

<details>
<summary><b>🧹 Aggressive Text Cleaning</b></summary>

Removes headers, signatures, quotes, and emails to avoid clustering on network topology (e.g. server domains) or frequent posters' names or email addresses. Ensures that clusters reflect **semantic content**, not metadata artifacts.

</details>

<details>
<summary><b>⚡ In-Memory, Cluster-Aware Cache</b></summary>

Implemented using only Python + NumPy — no external cache servers, no vector database for the cache layer. Cache is **partitioned by cluster**, bounding search to a small number of vectors. Uses cosine similarity with a **configurable threshold**:
- Higher threshold → high precision, lower recall
- Lower threshold → higher recall, more aggressive reuse of cached answers

</details>

<details>
<summary><b>🔁 State Management</b></summary>

Embedding model, GMM, cache, and basic stats are all initialized **once at startup** and reused per request. Avoids repeated heavy I/O and ensures consistent, low-latency behavior after warmup.

</details>

---

## 🔧 Configuration & Tuning

Current configuration lives directly in code:

| Parameter | Default | Location | Description |
|---|---|---|---|
| `n_clusters` | `15` | `SemanticClusterer` | Number of GMM macro-topic clusters |
| `threshold` | `0.65` | `ClusterAwareCache` in `main.py` | Cosine similarity threshold for cache hits |
| `max_size_per_cluster` | `100` | `ClusterAwareCache` default | LRU eviction limit per cluster bucket |

**Tuning tips:**
- Increase/decrease `n_clusters` to see how coarse/fine the macro-topics become
- Raise `threshold` for very precise matches; lower it if you can tolerate more semantic drift for speed

> **Future enhancement:** Moving these values to environment variables and adding persistence or a separate vector index for the cache if scale demands it.

---

## 🛠️ Local Utilities

| Script | Command | Purpose |
|---|---|---|
| `dataset.py` | `python dataset.py` | Test the cleaning pipeline and print sample cleaned documents |
| `cluster.py` | `python cluster.py` | Train GMM, profile clusters vs. original labels, deep dive into core vs. boundary documents |
| `view_dataset.py` | `python view_dataset.py` | Inspect raw Usenet documents from the tarball |

---

## ⚠️ Limitations & Considerations

> **Warmup time:** On first run (or container start), the service must load the embedding model, load and clean the dataset, and compute embeddings and train the GMM. This can take several minutes depending on hardware.

> **Memory footprint:** Embeddings + GMM + cache live in RAM. For very large datasets or higher-dimensional models, you may need more memory or to introduce sharding.

> **Dataset-specific:** This implementation is tuned for **20 Newsgroups**. Applying the same pipeline to very different domains may require different cleaning heuristics and/or a different number of clusters.

---

## 🤝 Contributing

Contributions and experiments are welcome. Some ideas:

- Adding new datasets or making the corpus pluggable
- Supporting configurable model backends (other sentence-transformer models)
- Exposing configuration via environment variables or a settings file
- Adding tests (unit + integration) for:
  - Text cleaning behavior
  - Cache hit/miss logic
  - API endpoints

If you plan to submit changes:

1. Fork the repository
2. Create a feature branch
3. Keep your changes focused and well-documented

---

## 🙏 Acknowledgements

This project is built around the **20 Newsgroups** dataset, originally curated and hosted by:

- **Tom Mitchell**, School of Computer Science, Carnegie Mellon University
- The **UCI KDD Archive**, University of California, Irvine

Please see the HTML files in `Dataset/` for the original dataset description, usage notes, and licensing terms. Use this material in accordance with the stated guidelines (educational use with appropriate attribution).

---

<div align="center">

*Built with FastAPI · BGE-Small · Gaussian Mixture Models · 20 Newsgroups*

</div>
