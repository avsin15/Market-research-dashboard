#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Builds a FAISS vector index of customer reviews
so we can retrieve the most relevant ones for Q&A.
"""

import os
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Paths
DATA_FILE = "data/sample/reviews_with_summaries.csv"
INDEX_FILE = "data/sample/review_index.faiss"
EMB_FILE = "data/sample/review_embeddings.npy"
META_FILE = "data/sample/review_metadata.csv"

def main():
    # Load dataset
    df = pd.read_csv(DATA_FILE)
    reviews = df["review_text"].fillna("").tolist()

    # Generate embeddings
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(reviews, show_progress_bar=True)

    # Save embeddings + metadata
    np.save(EMB_FILE, embeddings)
    df[["review_text", "topic", "sentiment"]].to_csv(META_FILE, index=False)

    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))

    faiss.write_index(index, INDEX_FILE)
    print(f"✅ Index built and saved to {INDEX_FILE}")

if __name__ == "__main__":
    main()


# In[ ]:




