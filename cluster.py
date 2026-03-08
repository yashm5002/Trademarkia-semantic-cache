import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.mixture import GaussianMixture

# Import our custom data loader from Part 1
from dataset import LocalDataLoader 

class SemanticClusterer:
    """
    PART 2: FUZZY CLUSTERING
    
    Design Justifications:
    1. Fuzzy vs. Hard Assignments: Gaussian Mixture Models (GMM) are used instead of 
       standard K-Means to provide a soft probability distribution. This explicitly 
       satisfies the requirement that documents belong to multiple topics to varying 
       degrees, rather than being forced into mutually exclusive bins.
    2. Number of Clusters (K=15): While the original dataset has 20 categories, many 
       are semantically redundant (e.g., 'comp.sys.ibm.pc.hardware' and 'comp.sys.mac.hardware'). 
       Reducing K to 15 forces the model to merge these artificial boundaries and learn 
       true semantic macro-topics, rather than overfitting to arbitrary human-made labels.
    """
    def __init__(self, n_clusters=15):
        # PART 1: EMBEDDING MODEL SELECTION
        # BAAI/bge-small-en-v1.5 was chosen deliberately over larger models or sparse 
        # lexical methods (BM25). It provides state-of-the-art semantic retrieval accuracy 
        # while remaining compact (~133MB) and highly performant for CPU-bound API environments. 
        # It ensures the system captures downstream semantic intent.
        print("Loading embedding model (BGE-Small)...")
        self.encoder = SentenceTransformer('BAAI/bge-small-en-v1.5')
        
        self.n_clusters = n_clusters
        self.gmm = GaussianMixture(n_components=n_clusters, covariance_type='diag', random_state=42)

    def train_model(self, clean_corpus):
        print(f"Embedding {len(clean_corpus)} documents. (This might take a couple of minutes...)")
        embeddings = self.encoder.encode(clean_corpus, show_progress_bar=True)
        
        print(f"Fitting Gaussian Mixture Model with {self.n_clusters} clusters...")
        self.gmm.fit(embeddings)
        
        print("Model training complete!")
        return embeddings

    def profile_clusters(self, embeddings, original_labels):
        """
        Validates the semantic meaningfulness of the clusters by profiling their 
        contents against the original dataset labels.
        """
        print("\n" + "="*60)
        print("PROFILING CLUSTERS TO REVEAL MACRO-TOPICS")
        print("="*60)
        predictions = self.gmm.predict(embeddings)
        
        cluster_profiles = {i: [] for i in range(self.n_clusters)}
        
        for label, cluster_id in zip(original_labels, predictions):
            cluster_profiles[cluster_id].append(label)
            
        for i in range(self.n_clusters):
            counts = Counter(cluster_profiles[i])
            top_3 = counts.most_common(3)
            total_in_cluster = len(cluster_profiles[i])
            
            print(f"\nCluster {i} ({total_in_cluster} docs) is mostly made of:")
            for category, count in top_3:
                percentage = (count / total_in_cluster) * 100 if total_in_cluster > 0 else 0
                print(f"  - {category}: {percentage:.1f}%")

    def deep_dive_analysis(self, clean_corpus, embeddings):
        """
        PART 2 REQUIREMENT: Convincing a sceptical reader.
        This exposes both the highly-certain cluster 'cores' and the genuinely 
        uncertain 'boundaries' where documents bleed across multiple semantic domains.
        """
        print("\n" + "="*60)
        print("DEEP DIVE: PROVING SEMANTIC MEANING TO THE SKEPTIC")
        print("="*60)
        
        probs = self.gmm.predict_proba(embeddings)
        
        margins = []
        top_choices = []
        second_choices = []
        
        for p in probs:
            sorted_indices = np.argsort(p)[::-1]
            top_idx = sorted_indices[0]
            second_idx = sorted_indices[1]
            
            margin = p[top_idx] - p[second_idx]
            margins.append(margin)
            top_choices.append((top_idx, p[top_idx]))
            second_choices.append((second_idx, p[second_idx]))
            
        margins = np.array(margins)
        
        target_cluster = 7
        print(f"\n[THE CORE] - Strongest matches for Cluster {target_cluster}:")
        
        core_indices = [i for i, (c, p) in enumerate(top_choices) if c == target_cluster and p > 0.99]
        
        for idx in core_indices[:2]: 
            print(f"\nConfidence: {top_choices[idx][1]*100:.1f}%")
            snippet = clean_corpus[idx][:250].replace('\n', ' ')
            print(f"Text: \"{snippet}...\"")

        print("\n\n[THE BOUNDARIES] - Documents where the AI is genuinely torn:")
        
        uncertain_indices = np.where(margins < 0.01)[0]
        
        for idx in uncertain_indices[:3]: 
            c1, p1 = top_choices[idx]
            c2, p2 = second_choices[idx]
            print(f"\nSplit Decision: {p1*100:.1f}% Cluster {c1} VS {p2*100:.1f}% Cluster {c2}")
            snippet = clean_corpus[idx][:300].replace('\n', ' ')
            print(f"Text: \"{snippet}...\"")

    def get_fuzzy_distribution(self, text):
        """
        Outputs a probability distribution across all clusters, rejecting hard assignments.
        """
        vec = self.encoder.encode([text])
        
        distribution = self.gmm.predict_proba(vec)[0]
        return distribution

# --- Run the Pipeline ---
if __name__ == "__main__":
    loader = LocalDataLoader()
    clean_data = loader.load_and_clean()
    
    clusterer = SemanticClusterer(n_clusters=15)
    
    embeddings = clusterer.train_model(clean_data)
    
    clusterer.profile_clusters(embeddings, loader.original_labels)
    
    clusterer.deep_dive_analysis(clean_data, embeddings)
    
    test_query = "Did God create the universe, or was it the Big Bang?"
    dist = clusterer.get_fuzzy_distribution(test_query)
    
    print("\n" + "="*60)
    print(f"TEST QUERY: '{test_query}'")
    print("Fuzzy Cluster Distribution:")
    
    top_indices = np.argsort(dist)[::-1][:3]
    for i in top_indices:
        print(f"Cluster {i}: {dist[i]*100:.1f}% match")
    print("="*60)