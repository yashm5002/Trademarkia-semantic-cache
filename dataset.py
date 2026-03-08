import tarfile
import re

class LocalDataLoader:
    """
    PART 1: EMBEDDING & VECTOR DATABASE SETUP (Data Preparation)
    
    Design Justifications for Corpus Preparation:
    The 20 Newsgroups dataset is notoriously noisy, filled with 1990s Usenet artifacts. 
    If fed directly into a dense vector embedding model (like BGE-Small), the model will 
    cluster documents based on network topology, frequent posters, or email domains 
    rather than the actual semantic intent of the text. 
    
    The deliberate choices below strip away this noise to ensure the downstream 
    clustering relies purely on semantic meaning.
    """
    def __init__(self, tar_path="Dataset/20_newsgroups.tar.gz"):
        self.tar_path = tar_path
        self.corpus = []
        self.original_labels = [] # Retained strictly for profiling/validation later

    def clean_text(self, raw_text):
        # 1. Strip the Header Block (Everything before the first double newline)
        # JUSTIFICATION: Headers contain routing paths, institution names, and 
        # timestamps. Without removal, downstream embeddings would falsely cluster 
        # documents originating from the same university server, bypassing topic semantics.
        parts = raw_text.split('\n\n', 1)
        body = parts[1] if len(parts) > 1 else raw_text
        
        # 2. Strip nested quotes and attribution lines
        # JUSTIFICATION: Usenet threads often quote the entirety of previous messages. 
        # Without stripping quotes, a 10-message debate becomes 10 nearly identical 
        # vectors in the database, breaking the semantic cache's ability to differentiate 
        # the unique contributions of each post.
        
        # Catches standard quotes (>), pipes (|>), dashed arrows (->, ->>), and initials (JD>)
        body = re.sub(r'^[ \t]*[-a-zA-Z]*[>|].*$', '', body, flags=re.MULTILINE)
        
        # Catches common Usenet attribution lines
        body = re.sub(r'^.*In article.*$', '', body, flags=re.MULTILINE)
        body = re.sub(r'^.*writes:.*$', '', body, flags=re.MULTILINE)
        body = re.sub(r'^.*says:.*$', '', body, flags=re.MULTILINE)
        
        # 3. Strip signatures
        # JUSTIFICATION: Heavy Usenet signatures (often containing ASCII art or quotes) 
        # create false, strong similarity across totally different topics written by 
        # the same prolific author. 
        body = re.sub(r'^--[\s\S]*$', '', body, flags=re.MULTILINE)
        
        # 4. Remove isolated email addresses, brackets, and normalize whitespace
        body = re.sub(r'\S+@\S+', '', body) # Removes leftover emails
        body = re.sub(r'<[^>]+>', '', body) # Removes leftover <brackets>
        body = re.sub(r'\s+', ' ', body).strip() # Squashes all whitespace into single spaces
        
        return body

    def load_and_clean(self):
        print(f"Reading directly from {self.tar_path} into memory...")
        
        with tarfile.open(self.tar_path, "r:gz") as tar:
            for member in tar:
                # We only want to process actual files, not the directory folders
                if member.isfile():
                    f = tar.extractfile(member)
                    if f is not None:
                        # Usenet files were latin-1 encoded
                        raw_text = f.read().decode('latin-1')
                        clean_body = self.clean_text(raw_text)
                        
                        # JUSTIFICATION: Drop documents that are too short after cleaning.
                        # A post that reduces to just "Me too." or "Thanks." acts as 
                        # semantic noise. Dense vector models require sufficient token 
                        # context to map meaning into 384-dimensional space. Documents 
                        # under 50 characters are actively detrimental to the clustering step.
                        if len(clean_body) > 50:
                            self.corpus.append(clean_body)
                            
                            # Extract the category name from the file path (e.g., 'alt.atheism')
                            category = member.name.split('/')[1]
                            self.original_labels.append(category)
                            
        print(f"Data pipeline complete. Retained {len(self.corpus)} clean documents.")
        return self.corpus

# --- Quick Test ---
if __name__ == "__main__":
    loader = LocalDataLoader()
    clean_data = loader.load_and_clean()
    
    # Let's peek at what the AI will actually see
    print("\nSample Cleaned Document:")
    print("========================")
    print(clean_data[0][:500] + "..." if len(clean_data[0]) > 500 else clean_data[0])