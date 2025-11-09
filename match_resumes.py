# match_resumes.py  (robust, numpy-only)
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader

EMB_DIR = "data/embeddings"
RESUME_DIR = "data/resumes"
TOP_K = 5
MODEL_NAME = "all-MiniLM-L6-v2"

# ---------------- helpers ----------------
def read_pdf_text(path):
    text_parts = []
    try:
        reader = PdfReader(path)
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text_parts.append(t)
    except Exception as e:
        print("  Warning: failed to read PDF", path, ":", e)
    return "\n".join(text_parts)

def read_resume_text(path):
    p = path.lower()
    if p.endswith(".pdf"):
        return read_pdf_text(path)
    # try common encodings fallback
    for enc in ("utf-8", "utf-16", "latin1", "cp1252"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except Exception:
            continue
    # fallback: read binary and decode ignoring errors
    with open(path, "rb") as f:
        raw = f.read()
    try:
        return raw.decode("utf-8", errors="ignore")
    except Exception:
        return raw.decode("latin1", errors="ignore")

# ---------------- load embeddings & model ----------------
if not os.path.exists(os.path.join(EMB_DIR, "jd_embeddings.npy")):
    raise SystemExit("Embeddings not found. Run embed_jds.py first.")

print("Loading embeddings and metadata...")
jd_embs = np.load(os.path.join(EMB_DIR, "jd_embeddings.npy"))   # shape (N, D)
jd_meta = pd.read_csv(os.path.join(EMB_DIR, "jd_metadata.csv"))
print("Loaded:", jd_embs.shape, "embeddings; metadata rows:", len(jd_meta))

print("Loading model:", MODEL_NAME)
model = SentenceTransformer(MODEL_NAME)

# ---------------- process resumes ----------------
os.makedirs(RESUME_DIR, exist_ok=True)
resume_files = [f for f in os.listdir(RESUME_DIR) if f.lower().endswith((".txt", ".pdf"))]

if not resume_files:
    print("No resumes found in", RESUME_DIR)
    print("Place .txt or .pdf resumes in that folder and re-run.")
    raise SystemExit

for fname in resume_files:
    path = os.path.join(RESUME_DIR, fname)
    print("\nProcessing resume:", fname)

    text = read_resume_text(path).strip()
    if not text:
        print("  Empty resume text; skipping.")
        continue

    # encode resume -> numpy vector
    q_emb = model.encode([text], convert_to_numpy=True)  # shape (1, D)
    # compute cosine similarities with all JD embeddings
    sims = cosine_similarity(q_emb, jd_embs)[0]          # shape (N,)

    # find top-k indices (as plain Python ints)
    top_idx = np.argsort(sims)[::-1][:TOP_K]            # numpy array of ints
    print(" Top {} matching job descriptions:".format(TOP_K))

    for rank, idx in enumerate(top_idx, start=1):
        idx = int(idx)                                   # ensure Python int
        score = float(sims[idx])
        # guard against missing columns
        jobtitle = jd_meta.iloc[idx].get("jobtitle", "N/A") if "jobtitle" in jd_meta.columns else "N/A"
        company = jd_meta.iloc[idx].get("company", "N/A") if "company" in jd_meta.columns else "N/A"
        snippet = str(jd_meta.iloc[idx].get("jobdescription", "") )[:200].replace("\n", " ")
        print(f" {rank}. {jobtitle} @ {company}  (score {score:.4f})")
        print("    snippet:", snippet)
    print("-" * 60)

print("\nAll resumes processed.")
