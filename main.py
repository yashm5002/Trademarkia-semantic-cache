from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import time

# Import our custom components
from dataset import LocalDataLoader
from cluster import SemanticClusterer
from cache import ClusterAwareCache

app = FastAPI(title="Semantic Clustering & Caching Service")

# --- GLOBAL STATE ---
# PART 4 JUSTIFICATION: State Management. 
# The embedding model (BGE-Small) and the trained Gaussian Mixture Model are 
# initialized globally on startup. This ensures that the heavy computational cost 
# of loading model weights into memory happens only once, rather than triggering a 
# massive I/O bottleneck on every individual POST /query request.
loader = LocalDataLoader()
clean_data = loader.load_and_clean()

clusterer = SemanticClusterer(n_clusters=15)
embeddings = clusterer.train_model(clean_data)

# PART 3 REQUIREMENT: The Tunable Decision
# Setting the threshold to 0.65 explicitly reveals that this system prioritizes 
# High Recall over strict Precision. A 0.65 boundary acts aggressively to catch 
# heavily paraphrased queries, assuming that users querying the same macro-cluster 
# likely benefit from the same cached conceptual distribution, drastically speeding up response times.
cache = ClusterAwareCache(num_clusters=15, threshold=0.65)

# PART 4 REQUIREMENT: Proper state management handled purely in Python memory,
# strictly avoiding external middleware like Redis.
stats = {
    "total_entries": 0,
    "hit_count": 0,
    "miss_count": 0
}

# --- MODELS ---
class QueryRequest(BaseModel):
    query: str

# --- ENDPOINTS ---

@app.post("/query")
async def process_query(request: QueryRequest):
    """
    PART 4: POST /query Endpoint.
    Embeds the natural language query, checks the custom semantic cache, 
    and returns the required JSON schema mapping hits/misses and dominant clusters.
    """
    user_query = request.query
    
    query_vector = clusterer.encoder.encode([user_query])[0]
    
    # Leverages the fuzzy GMM from Part 2 to determine the cache search boundary.
    distribution = clusterer.gmm.predict_proba([query_vector])[0]
    dominant_cluster = int(np.argmax(distribution))
    
    cached_hit = cache.check_cache(query_vector, dominant_cluster)
    
    if cached_hit:
        stats["hit_count"] += 1
        return {
            "query": user_query,
            "cache_hit": True,
            "matched_query": cached_hit["original_text"],
            "similarity_score": round(float(cached_hit["score"]), 4),
            "result": cached_hit["distribution"],
            "dominant_cluster": dominant_cluster
        }
    
    stats["miss_count"] += 1
    stats["total_entries"] += 1
    
    result_distribution = [round(float(p), 4) for p in distribution]
    
    cache_entry = {
        "original_text": user_query,
        "distribution": result_distribution,
        "score": 1.0 
    }
    # Enforces the Part 3 requirement of updating the bounded cache on a miss.
    cache.add_to_cache(user_query, query_vector, cache_entry, dominant_cluster)
    
    return {
        "query": user_query,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": None,
        "result": result_distribution,
        "dominant_cluster": dominant_cluster
    }

@app.get("/cache/stats")
async def get_stats():
    """
    PART 4: GET /cache/stats Endpoint.
    Returns the exact JSON schema required to monitor the in-memory cache performance.
    """
    total_requests = stats["hit_count"] + stats["miss_count"]
    hit_rate = (stats["hit_count"] / total_requests) if total_requests > 0 else 0
    
    return {
        "total_entries": stats["total_entries"],
        "hit_count": stats["hit_count"],
        "miss_count": stats["miss_count"],
        "hit_rate": round(hit_rate, 3)
    }

@app.delete("/cache")
async def flush_cache():
    """
    PART 4: DELETE /cache Endpoint.
    Flushes the cache entirely and resets all stateful statistics arrays.
    """
    cache.store = {i: {} for i in range(cache.n_clusters)}
    stats["total_entries"] = 0
    stats["hit_count"] = 0
    stats["miss_count"] = 0
    return {"message": "Cache flushed and stats reset."}