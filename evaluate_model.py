import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

EMB_PATH = "data/embeddings/jd_embeddings.npy"
META_PATH = "data/embeddings/jd_metadata.csv"
GT_PATH   = "data/ground_truth.csv"

TOP_K = 5

print("Loading embeddings & metadata...")
emb = np.load(EMB_PATH)
meta = pd.read_csv(META_PATH, encoding="latin1")
assert "uniq_id" in meta.columns, "uniq_id column missing in jd_metadata.csv"

id_series = meta["uniq_id"].astype(str)

print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading ground truth...")
gt = pd.read_csv(GT_PATH)

def evaluate_row(resume_text, true_id):
    # encode resume
    r_emb = model.encode(resume_text, normalize_embeddings=True)
    # cosine similarity
    sims = util.cos_sim(r_emb, emb)[0].cpu().numpy()
    top_idx = np.argsort(sims)[::-1][:TOP_K]
    top_ids = id_series.iloc[top_idx].tolist()
    top_scores = sims[top_idx].tolist()

    hit = str(true_id) in top_ids
    rank = top_ids.index(str(true_id)) + 1 if hit else None

    return {
        "top_ids": top_ids,
        "top_scores": top_scores,
        "hit": hit,
        "rank": rank
    }

rows = []
hits = 0
for i, r in gt.iterrows():
    res = evaluate_row(r["resume_text"], r["correct_jd_id"])
    hits += 1 if res["hit"] else 0
    rows.append({
        "resume_idx": i,
        "correct_jd_id": r["correct_jd_id"],
        "hit_in_top_k": res["hit"],
        "rank_if_hit": res["rank"],
        "top_ids": ";".join(res["top_ids"]),
        "top_scores": ";".join([f"{s:.4f}" for s in res["top_scores"]])
    })

# Metrics (single relevant JD per resume)
n = len(gt)
top1_acc = sum(1 for rr in rows if rr["rank_if_hit"] == 1) / n
acc_at_k = hits / n
precision_at_k = acc_at_k      # one relevant per query
recall_at_k = acc_at_k         # same reason
f1_at_k = acc_at_k if acc_at_k > 0 else 0.0

print("\n===== Evaluation (K = {}) =====".format(TOP_K))
print(f"Num resumes        : {n}")
print(f"Top-1 Accuracy     : {top1_acc:.3f}")
print(f"Accuracy@{TOP_K}   : {acc_at_k:.3f}")
print(f"Precision@{TOP_K}  : {precision_at_k:.3f}")
print(f"Recall@{TOP_K}     : {recall_at_k:.3f}")
print(f"F1@{TOP_K}         : {f1_at_k:.3f}")

# Save detailed preview
out = pd.DataFrame(rows)
out.to_csv("evaluation_topk_preview.csv", index=False)
print("\nSaved: evaluation_topk_preview.csv")

