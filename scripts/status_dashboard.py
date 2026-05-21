import streamlit as st
import pandas as pd
import faiss
import pickle
import os

st.set_page_config(page_title="IPC LegalGPT – System Status")

st.title("⚖️ IPC LegalGPT – System Status Dashboard")

# Paths
data_path = "../data"

# Step 1 check
validated_csv = os.path.join(data_path, "ipc_step1_validated.csv")
chunks_csv = os.path.join(data_path, "ipc_chunks.csv")
faiss_index_path = os.path.join(data_path, "faiss_index.bin")
metadata_path = os.path.join(data_path, "faiss_metadata.pkl")

st.subheader("📌 Pipeline Status")

if os.path.exists(validated_csv):
    df = pd.read_csv(validated_csv)
    st.success(f"STEP 1 Completed – {len(df)} IPC sections loaded")
else:
    st.error("STEP 1 not completed")

if os.path.exists(chunks_csv):
    chunks_df = pd.read_csv(chunks_csv)
    st.success(f"STEP 2 Completed – {len(chunks_df)} chunks created")
else:
    st.error("STEP 2 not completed")

if os.path.exists(faiss_index_path) and os.path.exists(metadata_path):
    index = faiss.read_index(faiss_index_path)
    st.success(f"STEP 3 Completed – {index.ntotal} embeddings stored in FAISS")
else:
    st.error("STEP 3 not completed")

st.markdown("---")
st.info("System is ready for Retrieval & Question Answering (STEP 4)")
