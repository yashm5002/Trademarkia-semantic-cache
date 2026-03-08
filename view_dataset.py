import tarfile

# Pointing exactly to where your file is in the screenshot
tar_path = "./Dataset/20_newsgroups.tar.gz"

def view_raw_tar_data():
    print(f"Cracking open the archive at {tar_path}...")
    
    try:
        # Open the tar.gz file for reading
        with tarfile.open(tar_path, "r:gz") as tar:
            c=0
            # Loop through the files inside the archive
            for member in tar:
                # We only want to read a file, not a folder
                if member.isfile():
                    print("\n" + "="*50)
                    print(f"CATEGORY/FILE: {member.name}")
                    print("="*50 + "\n")
                    
                    # Extract just this one file into memory
                    f = tar.extractfile(member)
                    if f is not None:
                        # 1990s internet text was usually latin-1 encoded, not utf-8
                        content = f.read().decode('latin-1')
                        print(content)
                        
                    print("\n" + "="*50)
                    print("END OF RAW DOCUMENT")
                    print("="*50)
                    c+=1
                    # We only want to see one, so we break the loop
                    if c==5:
                        break 
                    
    except FileNotFoundError:
        print(f"Oops! Couldn't find {tar_path}. Make sure you are running this from the root TRADEMARKIA folder.")

if __name__ == "__main__":
    view_raw_tar_data()