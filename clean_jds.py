# # clean_jds.py -- robust loader + cleaner for naukri CSV
# import os
# import pandas as pd
# import re
# import sys

# # 1) See where Python is running from (helps debug relative paths)
# print("Python working directory:", os.getcwd())
# print("Listing files in working dir:")
# for f in os.listdir(".")[:50]:
#     print(" ", f)
# print("-" * 40)

# # 2) Path to your CSV (update below if you prefer absolute path)
# rel_path = rel_path = rel_path = r"C:\Users\vkg25\OneDrive\Desktop\Anushka\IGDTUW Anushka\SEM3\AI LAB\AI_Project\data\archive\naukri_com-job_sample.csv"
# abs_path = os.path.abspath(rel_path)
# print("Trying to open (relative):", rel_path)
# print("Absolute path resolved to:", abs_path)
# print("-" * 40)

# # 3) Check file existence before reading
# if not os.path.exists(abs_path):
#     print("ERROR: CSV not found at that path.")
#     print("Make sure you saved the file at:", abs_path)
#     print("If the file is somewhere else, set 'rel_path' or use an absolute path.")
#     sys.exit(1)

# # 4) Try reading the CSV with safe parameters
# try:
#     # try default engine first with latin1 encoding (common for Kaggle dumps)
#     df = pd.read_csv(abs_path, encoding="latin1", low_memory=False)
#     print("CSV loaded successfully with encoding='latin1'.")
# except Exception as e:
#     print("First read_csv failed:", repr(e))
#     print("Trying with engine='python' and utf-8 fallback...")
#     try:
#         df = pd.read_csv(abs_path, engine="python", encoding="utf-8", low_memory=False)
#         print("CSV loaded successfully with engine='python' and utf-8.")
#     except Exception as e2:
#         print("Second attempt failed too:", repr(e2))
#         print("You may need to re-download the file or open it with Excel to inspect.")
#         sys.exit(1)

# # 5) Show columns and a quick preview
# print("Columns:", list(df.columns))
# print("Preview rows:")
# print(df.head(3))

# # 6) Extract and clean useful columns
# cols_needed = ["jobtitle", "jobdescription", "skills", "experience"]
# missing = [c for c in cols_needed if c not in df.columns]
# if missing:
#     print("WARNING: Some expected columns are missing:", missing)
#     # continue with whichever columns exist
# selected = [c for c in cols_needed if c in df.columns]
# jds = df[selected].dropna(subset=["jobdescription"]) if "jobdescription" in selected else df[selected].dropna()

# def clean_text(text):
#     text = re.sub(r"<[^>]+>", "", str(text))   # remove HTML tags
#     text = re.sub(r"\s+", " ", text).strip()   # collapse whitespace
#     return text

# if "jobdescription" in jds.columns:
#     jds["jobdescription"] = jds["jobdescription"].apply(clean_text)
# if "skills" in jds.columns:
#     jds["skills"] = jds["skills"].apply(clean_text)

# print("Total usable rows after dropna on jobdescription:", len(jds))

# # 7) Take a sample (100 or all if fewer)
# n_sample = 100
# if len(jds) > n_sample:
#     sample_jds = jds.sample(n_sample, random_state=42)
# else:
#     sample_jds = jds.copy()

# # 8) Save cleaned CSV and individual text files
# out_csv = "data/cleaned_job_descriptions.csv"
# out_folder = "data/job_descriptions"
# os.makedirs(os.path.dirname(out_csv), exist_ok=True)
# os.makedirs(out_folder, exist_ok=True)

# sample_jds.to_csv(out_csv, index=False, encoding="utf-8")
# print("Saved cleaned CSV to:", out_csv)

# for i, row in sample_jds.iterrows():
#     title = row.get("jobtitle", "") or "untitled"
#     safe_title = re.sub(r"[\\/*:?\"<>|]", "_", str(title))[:50]
#     fname = f"{i}_{safe_title}.txt"
#     path = os.path.join(out_folder, fname)
#     with open(path, "w", encoding="utf-8") as fout:
#         if "jobtitle" in row and pd.notna(row["jobtitle"]):
#             fout.write("TITLE: " + str(row["jobtitle"]) + "\n\n")
#         if "skills" in row and pd.notna(row["skills"]):
#             fout.write("SKILLS: " + str(row["skills"]) + "\n\n")
#         fout.write(str(row["jobdescription"]))
# print("Saved individual files to:", out_folder)

# print("DONE.")


# clean_jds.py -- robust loader + cleaner for Naukri CSV
import os
import pandas as pd
import re
import sys

print("=== START clean_jds.py ===")

# ----------------------------
# 1) CSV path setup
# ----------------------------
# Update this path if your dataset is in a different location
rel_path = r"data/archive/naukri_com-job_sample.csv"
abs_path = os.path.abspath(rel_path)
print("Trying to open (relative):", rel_path)
print("Absolute path resolved to:", abs_path)

# ----------------------------
# 2) Check file existence
# ----------------------------
if not os.path.exists(abs_path):
    print("❌ ERROR: CSV not found at:", abs_path)
    print("Make sure your Naukri dataset is in the path above.")
    sys.exit(1)
else:
    print("✅ Found CSV file:", abs_path, "size bytes:", os.path.getsize(abs_path))

# ----------------------------
# 3) Read CSV safely
# ----------------------------
try:
    df = pd.read_csv(abs_path, encoding="latin1", low_memory=False)
    print("✅ Loaded CSV successfully. Rows:", len(df))
except Exception as e:
    print("❌ Failed to read CSV:", repr(e))
    sys.exit(1)

# ----------------------------
# 4) Select and clean columns
# ----------------------------
cols_needed = ["jobtitle", "jobdescription", "skills", "experience"]
missing = [c for c in cols_needed if c not in df.columns]
if missing:
    print("⚠️ WARNING: Missing columns:", missing)

selected = [c for c in cols_needed if c in df.columns]
jds = df[selected].dropna(subset=["jobdescription"]) if "jobdescription" in selected else df[selected].dropna()
print("Loaded with columns:", list(jds.columns))

def clean_text(text):
    text = re.sub(r"<[^>]+>", "", str(text))  # remove HTML tags
    text = re.sub(r"\s+", " ", text).strip()  # collapse spaces
    return text

for col in ["jobdescription", "skills", "jobtitle"]:
    if col in jds.columns:
        jds[col] = jds[col].astype(str).apply(clean_text)

print("Usable rows after cleaning:", len(jds))

# ----------------------------
# 5) Take only 100 samples
# ----------------------------
n_sample = 17000
if len(jds) > n_sample:
    sample_jds = jds.sample(n_sample, random_state=42)
    print(f"📉 Sampled {n_sample} job descriptions from {len(jds)} total.")
else:
    sample_jds = jds.copy()
    print(f"Using all {len(jds)} job descriptions (less than {n_sample}).")

# ----------------------------
# 6) Save cleaned CSV and text files
# ----------------------------
out_csv = "data/cleaned_job_descriptions.csv"
out_folder = "data/job_descriptions"
os.makedirs(os.path.dirname(out_csv), exist_ok=True)
os.makedirs(out_folder, exist_ok=True)

sample_jds.to_csv(out_csv, index=False, encoding="utf-8")
print("✅ Saved cleaned CSV to:", out_csv)

for i, row in sample_jds.iterrows():
    title = row.get("jobtitle", "") or "untitled"
    safe_title = re.sub(r"[\\/*:?\"<>|]", "_", str(title))[:50]
    fname = f"{i}_{safe_title}.txt"
    path = os.path.join(out_folder, fname)
    with open(path, "w", encoding="utf-8") as fout:
        if "jobtitle" in row and pd.notna(row["jobtitle"]):
            fout.write("TITLE: " + str(row["jobtitle"]) + "\n\n")
        if "skills" in row and pd.notna(row["skills"]):
            fout.write("SKILLS: " + str(row["skills"]) + "\n\n")
        fout.write(str(row["jobdescription"]))
print("✅ Saved individual job description text files to:", out_folder)

print("=== DONE. Cleaned dataset with 100 JDs ready. ===")
