import streamlit as st
import pandas as pd
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="IPC LegalGPT", layout="wide")

st.title("⚖️ IPC LegalGPT")
st.caption("Ask questions based on Indian Penal Code (IPC)")

# -----------------------------
# Load resources
# -----------------------------
@st.cache_resource
def load_resources():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index("../data/faiss_index.bin")
    with open("../data/ipc_chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return model, index, chunks

model, index, chunks = load_resources()

# -----------------------------
# Chat history
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -----------------------------
# User input
# -----------------------------
query = st.chat_input("Ask your IPC legal question here...")

if query:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Embed query
    query_embedding = model.encode([query])
    D, I = index.search(query_embedding, k=3)

    # Build response
    response = "### Relevant IPC Sections:\n\n"
    for idx in I[0]:
        response += f"- {chunks[idx]}\n\n"

    # Show assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
