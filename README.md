# TradeMarkia – Semantic Clustering & Caching Service

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

This project is intentionally implemented **from first principles** (no Redis, no FAISS index for caching) to make the design decisions and trade-offs explicit.

---

## High-Level Architecture

At a high level, TradeMarkia runs the following pipeline:

1. **Data Preparation (`dataset.py`)**
   - Loads the **20 Newsgroups** corpus from `Dataset/20_newsgroups.tar.gz`.
   - Aggressively **cleans 1990s Usenet noise**:
     - Strips headers (routing paths, institution names, timestamps).
     - Removes nested quotes and attribution lines.
     - Removes signatures and email addresses.
     - Normalizes whitespace.
   - Drops documents that become too short after cleaning (≤ 50 characters) to avoid noisy, “content-free” embeddings.
   - Retains the original 20-newsgroup labels only for later profiling/validation.

2. **Semantic Embedding & Fuzzy Clustering (`cluster.py`)**
   - Uses `SentenceTransformer('BAAI/bge-small-en-v1.5')` as the embedding model.
   - Encodes each cleaned document into a dense vector.
   - Fits a **Gaussian Mixture Model (GMM)** with **15 clusters**:
     - 20 original categories → 15 macro-topics (merges semantically redundant newsgroups like similar hardware groups).
     - Provides **soft assignment** – each document (or query) has a **probability distribution across clusters**, not a single hard label.

3. **Cluster-Aware Semantic Cache (`cache.py`)**
   - Pure Python implementation, using:
     - `dict` + `OrderedDict` for per-cluster buckets and LRU eviction.
     - `numpy` for cosine similarity computations.
   - For each query:
     - Restricts search to **one cluster bucket** (O(K) rather than O(N) global).
     - Computes cosine similarity against cached vectors in that bucket.
     - Uses a **tunable similarity threshold** to decide hits vs. misses.
     - Maintains hits/misses and total entries for observability.

4. **FastAPI Service (`main.py`)**
   - On startup:
     - Loads and cleans the dataset (`LocalDataLoader`).
     - Trains the `SemanticClusterer` on the clean corpus.
     - Creates a `ClusterAwareCache` instance with a chosen similarity threshold.
   - Exposes HTTP endpoints:
     - `POST /query` – core semantic query + cache lookup.
     - `GET /cache/stats` – cache performance metrics.
     - `DELETE /cache` – flush cache + reset stats.

---

## Features

- **Semantic Query Understanding**
  - BGE-Small embeddings capture intent rather than surface-level keyword overlaps.

- **Fuzzy Topic Modeling**
  - GMM produces **probability distributions over macro-topics**, not just a single class.
  - Allows you to see **how “torn” the model is between topics** (e.g. religion vs. cosmology).

- **Cluster-Aware Semantic Cache**
  - Per-cluster buckets keep cache lookups bounded and efficient.
  - Uses **cosine similarity** over dense vectors with an explicit threshold.
  - **LRU eviction** per cluster to keep the cache size under control.

- **API-First Design**
  - `FastAPI` service ready to be called from web apps, CLI tools, or other services.
  - Includes **Dockerfile** and **docker-compose** for reproducible deployments.

- **Fully In-Memory State**
  - No external cache (Redis, Memcached) is used.
  - Embedding model, GMM, and cache all live inside the Python process.

---

## Project Structure

Key files:

- `main.py`  
  FastAPI application; wires together dataset loading, clustering, and the semantic cache. Hosts the `POST /query`, `GET /cache/stats`, and `DELETE /cache` endpoints.

- `dataset.py`  
  `LocalDataLoader` for reading and cleaning the 20 Newsgroups tarball.

- `cluster.py`  
  `SemanticClusterer` for embedding, training a GMM, profiling clusters, and producing fuzzy distributions for free-text queries.

- `cache.py`  
  `ClusterAwareCache`, a pure-Python, cluster-partitioned, LRU-like semantic cache over query embeddings.

- `view_dataset.py`  
  Utility to inspect raw Usenet documents from the `20_newsgroups.tar.gz` archive.

- `requirements.txt`  
  Python runtime dependencies.

- `Dockerfile`  
  Production-oriented container image for the FastAPI service.

- `docker-compose.yaml`  
  Compose setup that mounts the `Dataset` directory into the container.

- `Dataset/20newsgroups.html`, `Dataset/20newsgroups.data.html`  
  Original dataset metadata and licensing information from the UCI KDD Archive.

---

## Technology Stack

- **Language**: Python 3.10+
- **Web Framework**: FastAPI
- **Modeling**:
  - `sentence-transformers` (`BAAI/bge-small-en-v1.5`) for dense embeddings
  - `scikit-learn`’s `GaussianMixture` for fuzzy clustering
- **Numerical Computing**: NumPy
- **Infrastructure**:
  - Uvicorn for ASGI serving
  - Docker & docker-compose for containerization

---

## Getting Started (Local Development)

### 1. Prerequisites

- Python **3.10+**
- `git`
- (Optional) `virtualenv` or equivalent

### 2. Clone the Repository

```bash
git clone <your-repo-url> TradeMarkia
cd TradeMarkia
```

### 3. Create and Activate a Virtual Environment

On **Windows (PowerShell / cmd)**:

```bash
python -m venv venv
venv\Scripts\activate
```

On **macOS / Linux**:

```bash
python -m venv venv
source venv/bin/activate
```

### 4. Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install:

- `fastapi`
- `uvicorn`
- `sentence-transformers`
- `scikit-learn`
- `numpy`
- `pandas`
- `watchfiles`

### 5. Download / Place the Dataset

The service expects:

- `Dataset/20_newsgroups.tar.gz`

If it is not already present:

1. Create the `Dataset` directory:

   ```bash
   mkdir -p Dataset
   ```

2. Download the **20 Newsgroups** tarball from the UCI KDD Archive (see the HTML files in `Dataset/` for links), and save it as:

   ```text
   Dataset/20_newsgroups.tar.gz
   ```

### 6. Run the API with Uvicorn

From the project root:

```bash
uvicorn main:app --reload
```

Or, on Windows, you can use the helper script:

```bash
setup.bat
```

The service will start (by default) on `http://127.0.0.1:8000`.

You can then open the interactive docs:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

---

## Running with Docker

This repository includes a production-style Docker image and a docker-compose setup.

### 1. Build the Image

From the project root:

```bash
docker build -t trademarkia-service .
```

### 2. Run via `docker-compose`

Ensure that the `Dataset` directory (containing `20_newsgroups.tar.gz`) exists on your host at `./Dataset`.

Then run:

```bash
docker-compose up --build
```

This will:

- Build the image (if not already built).
- Start the FastAPI app at port **8000**.
- Mount `./Dataset` into `/app/Dataset` inside the container.

Access the API at:

- `http://localhost:8000/docs`

To stop:

```bash
docker-compose down
```

---

## API Reference

### `POST /query`

Process a natural-language query, perform fuzzy clustering, and consult the semantic cache.

**Request body:**

```json
{
  "query": "string"
}
```

**Example (curl):**

```bash
curl -X POST "http://127.0.0.1:8000/query" ^
  -H "Content-Type: application/json" ^
  -d "{\"query\": \"Did God create the universe, or was it the Big Bang?\"}"
```

**Example response (cache miss):**

```json
{
  "query": "Did God create the universe, or was it the Big Bang?",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": null,
  "result": [0.0012, 0.0345, ...],
  "dominant_cluster": 7
}
```

**Example response (cache hit):**

```json
{
  "query": "Did God create the universe, or was it the Big Bang?",
  "cache_hit": true,
  "matched_query": "Is the Big Bang compatible with religious beliefs?",
  "similarity_score": 0.9321,
  "result": [0.0008, 0.0123, ...],
  "dominant_cluster": 7
}
```

Where:

- **`result`** is the **fuzzy distribution** (length = number of clusters, default 15).
- **`dominant_cluster`** is the index of the cluster with the highest probability.
- **`cache_hit`**, **`matched_query`**, and **`similarity_score`** expose cache behavior.

---

### `GET /cache/stats`

Retrieve high-level cache performance metrics.

**Example:**

```bash
curl "http://127.0.0.1:8000/cache/stats"
```

**Response:**

```json
{
  "total_entries": 42,
  "hit_count": 30,
  "miss_count": 12,
  "hit_rate": 0.714
}
```

These stats are tracked in-process and reset when the service restarts or when the cache is flushed.

---

### `DELETE /cache`

Flushes all cache buckets and resets internal stats.

**Example:**

```bash
curl -X DELETE "http://127.0.0.1:8000/cache"
```

**Response:**

```json
{
  "message": "Cache flushed and stats reset."
}
```

---

## Design Decisions & Rationale

The codebase is heavily annotated with reasoning comments. Key decisions:

- **Embedding Model – BGE-Small (`BAAI/bge-small-en-v1.5`)**
  - Chosen as a compact (~133MB), high-quality sentence embedding model.
  - Well-suited for CPU-bound environments and production APIs.

- **Fuzzy Clustering with GMM**
  - `GaussianMixture` (with diagonal covariance) provides **soft cluster assignments**.
  - Reflects the reality that many documents lie at the boundary between topics.
  - Number of clusters (`n_components=15`) is **deliberately smaller** than the original 20 labels to merge redundant categories and discover macro-topics.

- **Aggressive Text Cleaning**
  - Removes headers, signatures, quotes, and emails to avoid clustering on:
    - Network topology (e.g., server domains).
    - Frequent posters’ names or email addresses.
  - Ensures that clusters reflect **semantic content**, not metadata artifacts.

- **In-Memory, Cluster-Aware Cache**
  - Implemented using only Python + NumPy:
    - No external cache servers.
    - No vector database for the cache layer.
  - Cache is **partitioned by cluster**, bounding search to a small number of vectors.
  - Uses cosine similarity with a **configurable threshold**:
    - Higher threshold → high precision, lower recall.
    - Lower threshold → higher recall, more aggressive reuse of cached answers.

- **State Management**
  - Embedding model, GMM, cache, and basic stats are all initialized **once at startup** and reused per request.
  - Avoids repeated heavy I/O and ensures consistent, low-latency behavior after warmup.

---

## Local Utilities

Beyond the API, there are scripts to help you inspect and understand the corpus and clustering:

- **`dataset.py` (`LocalDataLoader`)**
  - Run directly to test the cleaning pipeline and print sample cleaned documents:

    ```bash
    python dataset.py
    ```

- **`cluster.py` (`SemanticClusterer`)**
  - When run as a script, it:
    - Loads and cleans the corpus.
    - Trains the GMM.
    - Profiles clusters vs. original newsgroup labels.
    - Performs a “deep dive” analysis of core vs. boundary documents and prints illustrative examples.

    ```bash
    python cluster.py
    ```

- **`view_dataset.py`**
  - Opens the raw Usenet documents from the tarball and prints a handful to stdout (for debugging / inspection).

    ```bash
    python view_dataset.py
    ```

---

## Configuration & Tuning

Current configuration lives directly in code:

- **Number of clusters**: `n_clusters=15` (in `SemanticClusterer`).
- **Cache threshold**: `threshold=0.65` (set when instantiating `ClusterAwareCache` in `main.py`).
- **Max cache size per cluster**: `max_size_per_cluster=100` (default in `ClusterAwareCache`).

You can experiment with:

- Increasing/decreasing `n_clusters` to see how coarse/fine the macro-topics become.
- Raising/lowering the cache `threshold`:
  - If your application can tolerate more semantic “drift” for speed, lower it.
  - If you need very precise matches, raise it.

Future enhancements might include:

- Moving these values to environment variables.
- Adding persistence or a separate vector index for the cache if scale demands it.

---

## Limitations & Considerations

- **Warmup time**: On first run (or container start), the service must:
  - Load the embedding model.
  - Load and clean the dataset.
  - Compute embeddings and train the GMM.
  This can take several minutes depending on hardware.

- **Memory footprint**:
  - Embeddings + GMM + cache live in RAM.
  - For very large datasets or higher-dimensional models, you may need more memory or to introduce sharding.

- **Dataset-specific**:
  - This implementation is tuned for **20 Newsgroups**.
  - Applying the same pipeline to very different domains may require different cleaning heuristics and/or a different number of clusters.

---

## Contributing

Contributions and experiments are welcome. Some ideas:

- Adding new datasets or making the corpus pluggable.
- Supporting configurable model backends (other sentence-transformer models).
- Exposing configuration via environment variables or a settings file.
- Adding tests (unit + integration) for:
  - Text cleaning behavior.
  - Cache hit/miss logic.
  - API endpoints.

If you plan to submit changes:

- Fork the repository.
- Create a feature branch.
- Keep your changes focused and well-documented.

---

## Acknowledgements

This project is built around the **20 Newsgroups** dataset, originally curated and hosted by:

- **Tom Mitchell**, School of Computer Science, Carnegie Mellon University.
- The **UCI KDD Archive**, University of California, Irvine.

Please see the HTML files in `Dataset/` for the original dataset description, usage notes, and licensing terms. Use this material in accordance with the stated guidelines (educational use with appropriate attribution).

