# gradio_app.py
"""
Robust Gradio app for Resume -> Job Description matching.

Requirements:
  pip install sentence-transformers gradio scikit-learn numpy pandas PyPDF2
( PyPDF2 is optional if you don't upload PDF resumes. )
"""

import os
import tempfile
import traceback
import numpy as np
import pandas as pd
import gradio as gr

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# optional PDF reader
try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None

# Paths / settings
EMB_PATH = "data/embeddings/jd_embeddings.npy"
META_PATH = "data/embeddings/jd_metadata.csv"
MODEL_NAME = "all-MiniLM-L6-v2"   # small & fast; swap to all-mpnet-base-v2 for better accuracy
TOP_K_DEFAULT = 5


# ---------------------- helpers ----------------------
def try_read_csv(path):
    """Read CSV with robust encoding fallback."""
    for enc in ("utf-8", "latin1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    # final fallback (let pandas guess)
    return pd.read_csv(path, engine="python")


def read_pdf_text(path):
    """Extract text from a PDF file path using PyPDF2 (if installed)."""
    if PdfReader is None:
        return ""
    text_parts = []
    try:
        reader = PdfReader(path)
        for pg in reader.pages:
            t = pg.extract_text()
            if t:
                text_parts.append(t)
    except Exception:
        return ""
    return "\n".join(text_parts)


def load_resume_text(uploaded, pasted_text):
    """
    Return (text, error_message)
    - uploaded: might be None, a path string, or a gradio UploadedFile-like object
    - pasted_text: string from the textbox
    """
    # 1) If user uploaded a file, try to read it (uploaded may be a path string or file-like)
    if uploaded is not None:
        # Case A: Gradio sometimes passes a path string (NamedString)
        if isinstance(uploaded, str):
            path = uploaded
            try:
                if path.lower().endswith(".pdf"):
                    if PdfReader is None:
                        return None, "PyPDF2 not installed; can't read PDFs. Install: pip install PyPDF2"
                    return read_pdf_text(path), None
                # read bytes from file path
                with open(path, "rb") as bf:
                    bytes_data = bf.read()
            except Exception as e:
                return None, f"Failed to read uploaded file: {e}"

        else:
            # Case B: uploaded is object - try common attributes
            bytes_data = None
            # try attribute `.file`
            file_attr = getattr(uploaded, "file", None)
            if file_attr is not None:
                try:
                    file_attr.seek(0)
                    bytes_data = file_attr.read()
                except Exception:
                    bytes_data = None

            # try `.read()`
            if bytes_data is None:
                try:
                    bytes_data = uploaded.read()
                except Exception:
                    bytes_data = None

            # try `.name` path
            if bytes_data is None and getattr(uploaded, "name", None):
                try:
                    with open(uploaded.name, "rb") as bf:
                        bytes_data = bf.read()
                except Exception:
                    bytes_data = None

            if bytes_data is None:
                return None, "Couldn't read uploaded file (unsupported upload object)."

            # detect PDF by header bytes
            if bytes_data[:4] == b"%PDF":
                if PdfReader is None:
                    return None, "PyPDF2 not installed; can't read PDFs. Install: pip install PyPDF2"
                # write bytes to a temp file and extract text
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpf:
                    tmpf.write(bytes_data)
                    tmp_path = tmpf.name
                txt = read_pdf_text(tmp_path)
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
                return txt, None

        # decode bytes_data for text files with fallback encodings
        for enc in ("utf-8", "utf-16", "latin1", "cp1252"):
            try:
                s = bytes_data.decode(enc)
                return s, None
            except Exception:
                continue
        return bytes_data.decode("utf-8", errors="ignore"), None

    # 2) No uploaded file -> use pasted text
    if pasted_text is None or pasted_text.strip() == "":
        return None, "Please paste resume text or upload a resume (.txt or .pdf)."
    return pasted_text, None


def build_result_text(top_indices, scores, meta):
    """Format the top results into Markdown text."""
    parts = []
    for rank, idx in enumerate(top_indices, start=1):
        idx = int(idx)
        score = float(scores[idx])
        title = meta.iloc[idx].get("jobtitle", "N/A") if "jobtitle" in meta.columns else "N/A"
        company = meta.iloc[idx].get("company", "N/A") if "company" in meta.columns else "N/A"
        snippet = str(meta.iloc[idx].get("jobdescription", ""))[:800].replace("\n", " ")
        parts.append(f"**{rank}. {title}**  — {company}  \n**score:** `{score:.4f}`\n\n{snippet}\n\n---")
    return "\n\n".join(parts)


# ---------------------- load model & embeddings ----------------------
print("Loading model and embeddings... (this may take a few seconds)")
model = SentenceTransformer(MODEL_NAME)

if not os.path.exists(EMB_PATH):
    raise FileNotFoundError(f"Embeddings not found at {EMB_PATH}. Run your embedding script first.")

embeddings = np.load(EMB_PATH)  # shape (N, D)

# normalize embeddings to unit vectors (makes cosine similarity stable)
embeddings = normalize(embeddings, axis=1)

# load metadata with encoding fallbacks
if not os.path.exists(META_PATH):
    raise FileNotFoundError(f"Metadata CSV not found at {META_PATH}.")
meta = try_read_csv(META_PATH)

print("Loaded embeddings:", embeddings.shape, "metadata rows:", len(meta))


# ---------------------- endpoint ----------------------
def match_resume_endpoint(pasted_text, uploaded_file, top_k):
    """
    Gradio endpoint. Returns a markdown string or an error string.
    """
    try:
        resume_text, err = load_resume_text(uploaded_file, pasted_text)
        if err:
            return f"Error during matching:\n\n{err}"
        if not resume_text or resume_text.strip() == "":
            return "Error during matching:\n\nEmpty resume text after reading."

        # encode resume -> numpy vector (convert_to_numpy=True ensures numpy output)
        q_emb = model.encode([resume_text], convert_to_numpy=True)  # shape (1, D)
        # normalize
        q_emb = normalize(q_emb, axis=1)
        # compute cosine similarity with sklearn
        sims = cosine_similarity(q_emb, embeddings)[0]  # shape (N,)

        # top-K indices (descending)
        k = int(top_k) if top_k is not None else TOP_K_DEFAULT
        top_idx = np.argsort(sims)[::-1][:k]

        # build result text
        result_md = build_result_text(top_idx, sims, meta)
        return result_md

    except Exception as e:
        tb = traceback.format_exc()
        print("Error in match_resume_endpoint:\n", tb)
        return "Error during matching:\n\n" + str(e)


# ---------------------- UI ----------------------
with gr.Blocks() as demo:
    gr.Markdown("# AI Resume-to-Job Matcher")
    with gr.Row():
        txt = gr.Textbox(lines=12, label="Paste your resume text here")
        file_in = gr.File(file_types=[".pdf", ".txt"], label="Or upload resume (PDF or TXT)")
    with gr.Row():
        topk = gr.Slider(minimum=1, maximum=10, step=1, value=TOP_K_DEFAULT, label="Top K results")
    out = gr.Markdown()
    submit = gr.Button("Match")
    submit.click(fn=match_resume_endpoint, inputs=[txt, file_in, topk], outputs=out)

if __name__ == "__main__":
    # set share=True to create a public link (optional)
    demo.launch(share=True)

