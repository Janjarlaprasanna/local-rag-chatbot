"""Streamlit UI for the Local RAG Chatbot."""

import sys
from pathlib import Path

# Ensure src/ is on the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st

from rag.config import UPLOAD_DIR, OLLAMA_MODEL
from rag.document_loader import load_and_chunk, SUPPORTED_EXTENSIONS
from rag.vector_store import add_documents, list_sources, get_document_count, clear_collection
from rag.chain import ask_stream


st.set_page_config(page_title="Local RAG Chatbot", page_icon="📄", layout="wide")

# --- Session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar ---
with st.sidebar:
    st.header("📄 Document Manager")

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "txt", "csv"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button("Ingest Documents", type="primary", use_container_width=True):
            for uploaded_file in uploaded_files:
                save_path = UPLOAD_DIR / uploaded_file.name
                save_path.write_bytes(uploaded_file.getvalue())

                with st.spinner(f"Processing {uploaded_file.name}..."):
                    chunks = load_and_chunk(str(save_path))
                    count = add_documents(chunks)
                    st.success(f"{uploaded_file.name}: {count} chunks indexed")

    st.divider()

    # Status
    st.subheader("Status")
    sources = list_sources()
    doc_count = get_document_count()
    st.metric("Total Chunks", doc_count)

    if sources:
        st.write("**Indexed Sources:**")
        for source in sources:
            st.write(f"- {source}")
    else:
        st.info("No documents indexed yet. Upload files above.")

    st.divider()

    # Settings
    st.subheader("Settings")
    st.text(f"LLM: {OLLAMA_MODEL}")
    st.text(f"Embeddings: all-MiniLM-L6-v2")

    if st.button("Clear All Documents", use_container_width=True):
        clear_collection()
        st.rerun()

    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- Main chat area ---
st.title("🤖 Local RAG Chatbot")
st.caption("Ask questions about your uploaded documents. Fully local — no data leaves your machine.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📎 Sources"):
                for src in message["sources"]:
                    st.write(f"- {src}")

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Check if documents are loaded
    if get_document_count() == 0:
        response = "Please upload and ingest some documents first using the sidebar."
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
    else:
        # Stream assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            sources_placeholder = st.empty()
            full_response = ""
            sources = []

            try:
                for token in ask_stream(prompt):
                    if isinstance(token, dict):
                        # Final metadata
                        sources = token.get("sources", [])
                    else:
                        full_response += token
                        message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response)

                if sources:
                    with sources_placeholder.expander("📎 Sources"):
                        for src in sources:
                            st.write(f"- {src}")

            except Exception as e:
                full_response = f"Error connecting to Ollama. Make sure Ollama is running with `{OLLAMA_MODEL}` loaded.\n\nDetails: {e}"
                message_placeholder.markdown(full_response)

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": sources,
            })
