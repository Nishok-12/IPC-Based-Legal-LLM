import os
import pickle
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

# =========================
# PATH SETUP (SAFE)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

CSV_PATH = os.path.join(DATA_DIR, "ipc_clean.csv")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
CHUNKS_PATH = os.path.join(DATA_DIR, "ipc_chunks.pkl")

# =========================
# LOAD DATA
# =========================
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")

df = pd.read_csv(CSV_PATH, encoding="latin-1")


# Change column name if needed
TEXT_COLUMN = df.columns[0]

texts = df[TEXT_COLUMN].dropna().astype(str).tolist()

print(f"Loaded {len(texts)} text records")

# =========================
# LOAD EMBEDDING MODEL
# =========================
model = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# CREATE EMBEDDINGS
# =========================
embeddings = model.encode(
    texts,
    show_progress_bar=True,
    convert_to_numpy=True
)

# Normalize for cosine similarity
faiss.normalize_L2(embeddings)

dimension = embeddings.shape[1]

# =========================
# CREATE FAISS INDEX
# =========================
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

print("FAISS index created")

# =========================
# SAVE FILES (IMPORTANT)
# =========================
faiss.write_index(index, FAISS_INDEX_PATH)

with open(CHUNKS_PATH, "wb") as f:
    pickle.dump(texts, f)

print("===================================")
print("STEP 3 COMPLETED SUCCESSFULLY ✅")
print("Saved files:")
print(" -", FAISS_INDEX_PATH)
print(" -", CHUNKS_PATH)
print("===================================")
