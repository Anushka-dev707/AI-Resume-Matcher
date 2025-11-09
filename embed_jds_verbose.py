# # embed_jds_verbose.py
# import os, sys, time
# import numpy as np
# import pandas as pd

# print("START embed_jds_verbose.py")
# start_time = time.time()

# # ----------------------------
# # Basic checks and paths
# # ----------------------------
# CLEAN_CSV = "data/cleaned_job_descriptions.csv"
# EMB_DIR = "data/embeddings"
# EMB_FILE = os.path.join(EMB_DIR, "jd_embeddings.npy")
# META_FILE = os.path.join(EMB_DIR, "jd_metadata.csv")

# print("Working dir:", os.getcwd())
# print("Checking files...")

# if not os.path.exists(CLEAN_CSV):
#     print("ERROR: cleaned CSV not found at:", CLEAN_CSV)
#     print("Make sure you ran clean_jds.py and saved cleaned_job_descriptions.csv in data/")
#     sys.exit(1)
# else:
#     print("Found cleaned CSV:", CLEAN_CSV, "size bytes:", os.path.getsize(CLEAN_CSV))

# os.makedirs(EMB_DIR, exist_ok=True)

# # ----------------------------
# # Load cleaned JDs
# # ----------------------------
# try:
#     df = pd.read_csv(CLEAN_CSV, encoding="utf-8")
#     print("Loaded cleaned CSV. Rows:", len(df), "Columns:", list(df.columns))
# except Exception as e:
#     print("ERROR reading CSV:", repr(e))
#     sys.exit(1)

# if "jobdescription" not in df.columns:
#     print("ERROR: 'jobdescription' column not present in CSV.")
#     sys.exit(1)

# texts = df["jobdescription"].fillna("").astype(str).tolist()

# # ----------------------------
# # Import model & encode
# # ----------------------------
# try:
#     from sentence_transformers import SentenceTransformer
# except Exception as e:
#     print("ERROR: sentence-transformers not installed or import failed:", repr(e))
#     print("Run: pip install sentence-transformers")
#     sys.exit(1)

# MODEL_NAME = "all-MiniLM-L6-v2"
# print("Loading model:", MODEL_NAME, " — this may take a while the first time (downloads ~50-200MB)")
# t0 = time.time()
# try:
#     model = SentenceTransformer(MODEL_NAME)
# except Exception as e:
#     print("ERROR while loading model:", repr(e))
#     print("Check your internet connection and that sentence-transformers is installed.")
#     sys.exit(1)
# print("Model loaded in {:.1f}s".format(time.time() - t0))

# # quick test encode of one short string
# try:
#     _ = model.encode(["test"], convert_to_numpy=True)
#     print("Sanity encode OK.")
# except Exception as e:
#     print("ERROR during small encode test:", repr(e))
#     sys.exit(1)

# # Encode all texts in batches with progress printing
# batch_size = 128
# emb_list = []
# print("Encoding {} texts in batches of {}...".format(len(texts), batch_size))
# t0 = time.time()
# for i in range(0, len(texts), batch_size):
#     batch = texts[i : i + batch_size]
#     emb = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
#     emb_list.append(emb)
#     print(f"  encoded {i + len(batch)}/{len(texts)}")
# emb = np.vstack(emb_list)
# print("Finished encoding. Embeddings shape:", emb.shape)
# print("Encoding time: {:.1f}s".format(time.time() - t0))

# # ----------------------------
# # Save embeddings & metadata
# # ----------------------------
# np.save(EMB_FILE, emb)
# df.to_csv(META_FILE, index=False, encoding="utf-8")
# print("Saved embeddings to:", EMB_FILE, "size:", os.path.getsize(EMB_FILE))
# print("Saved metadata to:", META_FILE, "size:", os.path.getsize(META_FILE))

# # ----------------------------
# # Small similarity demo
# # ----------------------------
# try:
#     from sklearn.metrics.pairwise import cosine_similarity
# except Exception as e:
#     print("WARNING: scikit-learn not installed. Install with: pip install scikit-learn")
#     print("Skipping similarity demo.")
# else:
#     sample_resume = "Experienced Python developer with ML and NLP experience, using PyTorch and FastAPI for deployments."
#     q_emb = model.encode([sample_resume], convert_to_numpy=True)
#     sims = cosine_similarity(q_emb, emb)[0]
#     top = np.argsort(sims)[::-1][:5]
#     print("\nTop-5 matches (similarity scores):")
#     for rank, idx in enumerate(top, start=1):
#         title = df.iloc[idx].get("jobtitle", "N/A")
#         score = sims[idx]
#         print(f"{rank}. idx={idx}  score={score:.4f}  title={title}")

# total_time = time.time() - start_time
# print("\nALL DONE. Total script time: {:.1f}s".format(total_time))

# embed_jds_verbose.py (patched)
import os
import sys
import time
import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap

print("START embed_jds_verbose.py")
start_time = time.time()

# ----------------------------
# Basic checks and paths
# ----------------------------
CLEAN_CSV = "data/cleaned_job_descriptions.csv"
EMB_DIR = "data/embeddings"
EMB_FILE = os.path.join(EMB_DIR, "jd_embeddings.npy")
META_FILE = os.path.join(EMB_DIR, "jd_metadata.csv")
TEMP_EMB_FILE = os.path.join(EMB_DIR, "jd_embeddings.tmp.npy")

print("Working dir:", os.getcwd())
print("Checking files...")

if not os.path.exists(CLEAN_CSV):
    print("ERROR: cleaned CSV not found at:", CLEAN_CSV)
    print("Make sure you ran clean_jds.py and saved cleaned_job_descriptions.csv in data/")
    sys.exit(1)
else:
    print("Found cleaned CSV:", CLEAN_CSV, "size bytes:", os.path.getsize(CLEAN_CSV))

os.makedirs(EMB_DIR, exist_ok=True)

# ----------------------------
# Load cleaned JDs
# ----------------------------
try:
    # use latin1 for robust reading of older Kaggle csvs with weird chars
    df = pd.read_csv(CLEAN_CSV, encoding="latin1")
    print("Loaded cleaned CSV. Rows:", len(df), "Columns:", list(df.columns))
except Exception as e:
    print("ERROR reading CSV:", repr(e))
    sys.exit(1)

if "jobdescription" not in df.columns:
    print("ERROR: 'jobdescription' column not present in CSV.")
    sys.exit(1)

# fill missing and ensure string type
texts = df["jobdescription"].fillna("").astype(str).tolist()
n_texts = len(texts)
print("Num texts to encode:", n_texts)

# ----------------------------
# Import model & encode
# ----------------------------
try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    print("ERROR: sentence-transformers not installed or import failed:", repr(e))
    print("Run: pip install sentence-transformers")
    sys.exit(1)

MODEL_NAME = "all-MiniLM-L6-v2"
print("Loading model:", MODEL_NAME, " — this may take a while the first time (downloads ~50-200MB)")
t0 = time.time()
try:
    model = SentenceTransformer(MODEL_NAME)
except Exception as e:
    print("ERROR while loading model:", repr(e))
    print("Check your internet connection and that sentence-transformers is installed.")
    sys.exit(1)
print("Model loaded in {:.1f}s".format(time.time() - t0))

# quick test encode of one short string
try:
    _ = model.encode(["test"], convert_to_numpy=True)
    print("Sanity encode OK.")
except Exception as e:
    print("ERROR during small encode test:", repr(e))
    sys.exit(1)

# ----------------------------
# Prepare memmap for final embeddings (use open_memmap so .npy header is written)
# ----------------------------
emb_dim = model.get_sentence_embedding_dimension()
print("Embedding dimension:", emb_dim)

# if embedding file already exists and matches shape, we avoid re-computing
existing = None
if os.path.exists(EMB_FILE):
    try:
        existing = np.load(EMB_FILE, mmap_mode="r")
        if existing.shape[0] == n_texts and existing.shape[1] == emb_dim:
            print("Existing embeddings found with correct shape; skipping encoding.")
            emb = existing  # read-only memmap
        else:
            print("Existing embeddings present but shape differs. Will (re)create embeddings.")
            existing = None
    except Exception:
        # cannot load (corrupt) -> recreate
        existing = None

if existing is None:
    # remove any stale temp file
    if os.path.exists(TEMP_EMB_FILE):
        print("Found temp embedding file from previous run. Removing and starting fresh.")
        try:
            os.remove(TEMP_EMB_FILE)
        except Exception:
            pass

    print("Creating memmap (valid .npy):", TEMP_EMB_FILE, "shape=({}, {})".format(n_texts, emb_dim))
    emb_mem = open_memmap(TEMP_EMB_FILE, mode="w+", dtype=np.float32, shape=(n_texts, emb_dim))

    # Encode all texts in batches with progress printing
    batch_size = 128  # lower if memory is tight
    print("Encoding {} texts in batches of {}...".format(n_texts, batch_size))
    t0 = time.time()
    for i in range(0, n_texts, batch_size):
        batch_texts = texts[i : i + batch_size]
        emb_batch = model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
        emb_mem[i : i + len(emb_batch), :] = emb_batch
        emb_mem.flush()
        print(f"  encoded {i + len(emb_batch)}/{n_texts}")
    total_encode_time = time.time() - t0
    print("Finished encoding. Total encoding time: {:.1f}s".format(total_encode_time))

    # close memmap and move into final .npy atomically
    del emb_mem
    try:
        if os.path.exists(EMB_FILE):
            os.remove(EMB_FILE)
        os.replace(TEMP_EMB_FILE, EMB_FILE)
        print("Saved embeddings memmap to:", EMB_FILE)
    except Exception as e:
        print("ERROR saving final embeddings:", repr(e))
        if os.path.exists(TEMP_EMB_FILE):
            print("Temp embedding file at:", TEMP_EMB_FILE)
        sys.exit(1)

# load final embeddings (read-only memmap)
try:
    emb = np.load(EMB_FILE, mmap_mode="r")
    print("Loaded embeddings with mmap_mode='r'. shape:", getattr(emb, "shape", None), "dtype:", getattr(emb, "dtype", None))
except Exception as e:
    print("ERROR loading embeddings file:", repr(e))
    print("Deleting corrupted embeddings file so it can be re-created on next run.")
    try:
        os.remove(EMB_FILE)
    except Exception:
        pass
    sys.exit(1)

print("Embeddings shape (on disk):", emb.shape, "dtype:", emb.dtype)

# ----------------------------
# Save metadata (ensure uniq_id)
# ----------------------------
if "uniq_id" not in df.columns:
    df["uniq_id"] = df.index.astype(str)
else:
    df["uniq_id"] = df["uniq_id"].astype(str)

try:
    df.to_csv(META_FILE, index=False, encoding="utf-8")
    print("Saved metadata to:", META_FILE, "size:", os.path.getsize(META_FILE))
except Exception as e:
    print("WARNING: failed to save metadata as utf-8, trying latin1:", repr(e))
    df.to_csv(META_FILE, index=False, encoding="latin1")
    print("Saved metadata to:", META_FILE, "size:", os.path.getsize(META_FILE))

# ----------------------------
# Small similarity demo
# ----------------------------
try:
    from sklearn.metrics.pairwise import cosine_similarity
except Exception as e:
    print("WARNING: scikit-learn not installed. Install with: pip install scikit-learn")
    print("Skipping similarity demo.")
else:
    sample_resume = "Experienced Python developer with ML and NLP experience, using PyTorch and FastAPI for deployments."
    q_emb = model.encode([sample_resume], convert_to_numpy=True, normalize_embeddings=True)
    emb_arr = np.array(emb) if isinstance(emb, np.memmap) else emb
    sims = cosine_similarity(q_emb, emb_arr)[0]
    top = np.argsort(sims)[::-1][:5]
    print("\nTop-5 matches (similarity scores):")
    for rank, idx in enumerate(top, start=1):
        title = df.iloc[idx].get("jobtitle", "N/A")
        score = sims[idx]
        print(f"{rank}. idx={idx}  score={score:.4f}  title={title}")

total_time = time.time() - start_time
print("\nALL DONE. Total script time: {:.1f}s".format(total_time))
