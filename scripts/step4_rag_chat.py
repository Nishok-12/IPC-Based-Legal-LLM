import streamlit as st
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
import os

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="IPC LegalGPT", layout="wide")
st.title("⚖️ IPC LegalGPT – RAG Chat")
st.caption("Ask questions based on Indian Penal Code (IPC)")

# -----------------------------
# Load Model & FAISS
# -----------------------------
@st.cache_resource
def load_all():
    model = SentenceTransformer("all-MiniLM-L6-v2")

    index = faiss.read_index("../data/faiss_index.bin")

    with open("../data/faiss_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    return model, index, metadata

model, index, metadata = load_all()

# -----------------------------
# Chat History
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -----------------------------
# User Input
# -----------------------------
query = st.chat_input("Ask a legal question (e.g., punishment for theft)")

if query:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Embed the query
    query_vector = model.encode([query]).astype("float32")

    # FAISS search
    D, I = index.search(query_vector, k=3)

    # Build answer
    answer = "### 🔍 Relevant IPC Sections\n\n"

    for rank, idx in enumerate(I[0], start=1):
        section = metadata[idx]
        section_no = section.get("section_number", "N/A")
        text = section.get("chunk_text", "")

        answer += f"""
### 📘 IPC Section {section_no}
{text}

---
"""


    # Show assistant response
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
