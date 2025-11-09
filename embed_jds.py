# embed_jds.py
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# File paths
# ----------------------------
CLEAN_CSV = "data/cleaned_job_descriptions.csv"  # From your previous script
EMB_DIR = "data/embeddings"
os.makedirs(EMB_DIR, exist_ok=True)

# ----------------------------
# Load cleaned JDs
# ----------------------------
df = pd.read_csv(CLEAN_CSV, encoding="utf-8")
print(f"Loaded {len(df)} job descriptions.")

texts = df["jobdescription"].fillna("").astype(str).tolist()

# ----------------------------
# Load model
# ----------------------------
MODEL_NAME = "all-MiniLM-L6-v2"  # smaller, faster alternative to mpnet
print("Loading model:", MODEL_NAME)
model = SentenceTransformer(MODEL_NAME)

# ----------------------------
# Encode job descriptions
# ----------------------------
embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
print("Embeddings shape:", embeddings.shape)

# Save embeddings and metadata
np.save(os.path.join(EMB_DIR, "jd_embeddings.npy"), embeddings)
df.to_csv(os.path.join(EMB_DIR, "jd_metadata.csv"), index=False)
print("Saved embeddings and metadata to", EMB_DIR)

# ----------------------------
# Quick test: similarity with a sample resume
# ----------------------------
resume_text = """Experienced Python developer with knowledge of data analysis, 
machine learning, and model deployment using Streamlit and FastAPI. 
Worked on NLP projects and have hands-on experience with PyTorch."""

resume_emb = model.encode([resume_text], convert_to_numpy=True)
similarities = cosine_similarity(resume_emb, embeddings)[0]

# Get top 5 matches
top_indices = np.argsort(similarities)[::-1][:5]
print("\nTop 5 matching job descriptions:")
for i, idx in enumerate(top_indices, start=1):
    title = df.iloc[idx].get("jobtitle", "N/A")
    company = df.iloc[idx].get("company", "N/A") if "company" in df.columns else "N/A"
    snippet = df.iloc[idx]["jobdescription"][:200].replace("\n", " ")
    print(f"{i}. {title} @ {company}")
    print(f"   Similarity: {similarities[idx]:.4f}")
    print(f"   Snippet: {snippet}...\n")

print("Saved embeddings and metadata to", EMB_DIR)

