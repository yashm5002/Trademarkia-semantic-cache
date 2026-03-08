import numpy as np
from collections import OrderedDict

class ClusterAwareCache:
    """
    PART 3: SEMANTIC CACHE LAYER
    
    Design Justifications:
    1. First-Principles Implementation: Built entirely using native Python structures 
       (dict, OrderedDict) and NumPy. This deliberately avoids external dependencies 
       like Redis or Memcached, fulfilling the strict "built from scratch" constraint.
    2. Cluster-Aware Efficiency: Instead of a flat cache that requires comparing a new 
       query against every cached item (O(N) complexity), this cache is partitioned 
       by cluster. By leveraging the fuzzy clustering output from Part 2, the cache 
       search space is strictly bounded to max_size_per_cluster.
    3. Tunable Heuristic: The 'threshold' parameter defines the boundary between a 
       cache miss and a hit.
    """
    def __init__(self, num_clusters=15, max_size_per_cluster=100, threshold=0.75):
        self.num_clusters = num_clusters
        self.max_size_per_cluster = max_size_per_cluster
        
        # TUNABLE DECISION: The Cosine Similarity Threshold
        # This explicit value dictates system behaviour: 
        # - A high threshold (e.g., 0.85) creates a strict, high-precision cache that 
        #   rarely serves incorrect answers but suffers a lower hit rate.
        # - A low threshold (e.g., 0.60) creates a loose, high-recall cache that acts 
        #   aggressively, revealing that the system values speed over semantic nuance.
        self.threshold = threshold
        
        # Native implementation of an LRU (Least Recently Used) cache mechanism.
        self.store = {i: OrderedDict() for i in range(num_clusters)}
        
        self.hits = 0
        self.misses = 0

    def cosine_similarity(self, vec1, vec2):
        """
        Calculates the semantic distance between two queries.
        Because the search space is bounded by the cluster size (<= 100), 
        NumPy dot product operations are computationally cheaper and faster 
        than initializing a heavy vector database index (like FAISS) for cache lookups.
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    def check_cache(self, query_vector, cluster_id):
        """
        Evaluates a query against previously seen queries, recognising identical 
        intents even if phrasing differs. By strictly searching within the assigned 
        cluster_id, cache scaling issues are entirely mitigated.
        """
        bucket = self.store.get(cluster_id)
        
        if bucket is None or not bucket:
            self.misses += 1
            # Added flush=True to ensure logs appear immediately in Docker
            print(f"\n[CACHE LOG] Cluster {cluster_id} is currently empty. Standard Miss.", flush=True)
            return None

        best_match_text = None
        best_score = -1.0
        best_response = None

        for cached_text, cache_data in bucket.items():
            score = self.cosine_similarity(query_vector, cache_data["vector"])
            
            if score > best_score:
                best_score = score
                best_match_text = cached_text
                best_response = cache_data["response"]

        if best_score >= self.threshold:
            self.hits += 1
            # Maintain the LRU invariant by marking this hit as the most recently used.
            self.store[cluster_id].move_to_end(best_match_text)
            print(f"\n[CACHE HIT \u2705] Similarity Score: {best_score:.4f}", flush=True)
            print(f"[CACHE HIT \u2705] Matched with cached query: '{best_match_text}'", flush=True)
            return best_response

        self.misses += 1
        print(f"\n[SMART MISS \u26A0\uFE0F] Best match in Cluster {cluster_id} was: '{best_match_text}'", flush=True)
        print(f"[SMART MISS \u26A0\uFE0F] Similarity Score was {best_score:.4f}, which is below the threshold of {self.threshold}.", flush=True)
        return None

    def add_to_cache(self, query_text, query_vector, response, cluster_id):
        """
        Inserts new items into the cluster-specific cache.
        If the bounded space is full, it enforces an O(1) eviction policy.
        """
        bucket = self.store[cluster_id]
        
        bucket[query_text] = {
            "vector": query_vector,
            "response": response
        }
        
        # O(1) LRU Eviction: Dictates that when semantic capacity is reached, 
        # the system forgets the least queried information first, handled natively.
        if len(bucket) > self.max_size_per_cluster:
            bucket.popitem(last=False)

    def get_stats(self):
        """
        Outputs state management metrics, mapped directly to Part 4 
        GET /cache/stats endpoint requirements.
        """
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests) if total_requests > 0 else 0.0
        
        total_entries = sum(len(bucket) for bucket in self.store.values())
        
        return {
            "total_entries": total_entries,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(hit_rate, 4)
        }

    def clear_cache(self):
        """
        Provides cache flushing capabilities as requested by the DELETE /cache endpoint.
        """
        self.store = {i: OrderedDict() for i in range(self.num_clusters)}
        self.hits = 0
        self.misses = 0