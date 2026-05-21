import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="IPC LegalGPT (Ollama RAG)", layout="wide")
st.title("⚖️ IPC LegalGPT ")
st.caption("Local LLaMA-powered Legal Assistant")

# -----------------------------
# LOAD EMBEDDINGS + FAISS
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
# CHAT HISTORY
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -----------------------------
# USER INPUT
# -----------------------------
query = st.chat_input("Ask a legal question...")

if query:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # -----------------------------
    # RAG RETRIEVAL
    # -----------------------------
    query_vector = model.encode([query]).astype("float32")
    D, I = index.search(query_vector, k=3)

    context = ""
    for idx in I[0]:
        section = metadata[idx]
        section_no = section.get("section_number", "N/A")
        text = section.get("chunk_text", "")
        context += f"IPC Section {section_no}:\n{text}\n\n"

    # -----------------------------
    # OLLAMA CALL
    # -----------------------------
    response = ollama.chat(
        model="llama3",
        messages=[
        {
            "role": "system",
            "content": """
You are LegalGPT, an AI legal assistant specialized in the Indian Penal Code.

Always respond in the following structured format:

📜 Section Number:
IPC Section <number>

📌 Title:
<Title of the section>

📖 Explanation:
<Explain the legal meaning clearly>

⚖️ Punishment:
<Explain punishment>

📊 Legal Classification
Field | Information
Cognizable | Yes/No
Bailable | Yes/No
Triable By | Court authority

🧾 Example Scenario
Give a simple real-life example where this section applies.

📚 Related IPC Sections
Mention 3–4 related IPC sections with short explanation.

💡 Suggested Questions
Suggest 3–5 follow-up legal questions.

⚠️ Disclaimer
This information is for educational purposes and should not replace professional legal advice.

Rules:
- Use only the provided IPC context.
- Do not greet the user.
- Provide clear and structured legal information.
"""
        },
        {
            "role": "user",
            "content": f"""
Context:
{context}

Question:
{query}
"""
        }
    ],
    options={
        "temperature": 0.2,
        "num_predict": 600
    }
)

    answer = response["message"]["content"]
    # -----------------------------
    # DISPLAY RESPONSE
    # -----------------------------
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
