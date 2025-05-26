import os
import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from ibm_watsonx_ai.foundation_models import ModelInference

from dotenv import load_dotenv
load_dotenv()

# Config
CHROMA_COLLECTION_NAME = "support_kb"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)

@st.cache_resource
def init_chroma():
    chroma_client = chromadb.Client()
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL_NAME)
    try:
        return chroma_client.get_collection(name=CHROMA_COLLECTION_NAME)
    except:
        return chroma_client.create_collection(name=CHROMA_COLLECTION_NAME, embedding_function=embedding_fn)

import os
import streamlit as st
from ibm_watsonx_ai.foundation_models import ModelInference
from dotenv import load_dotenv

load_dotenv()  # Load env vars from .env file (if using one)

@st.cache_resource
def get_watson_model():
    api_key = os.getenv("WATSONX_API_KEY")
    project_id = os.getenv("WATSONX_PROJECT_ID")
    url = os.getenv("WATSONX_URL")

    if not api_key or not project_id or not url:
        raise ValueError("Missing WATSONX_API_KEY, WATSONX_PROJECT_ID, or WATSONX_URL.")

    return ModelInference(
        model_id="ibm/granite-3-3-8b-instruct",
        project_id=project_id,
        credentials={
            "apikey": api_key,
            "url": url
        }
    )
st.set_page_config(page_title="SmartResolve (Chroma)", layout="centered")
st.title("🤖 SmartResolve - AI Support Assistant (Chroma + Watsonx)")

with st.sidebar:
    st.header("📄 Upload Knowledge Base")
    uploaded_file = st.file_uploader("Upload .txt file", type="txt")
    if uploaded_file:
        lines = uploaded_file.read().decode("utf-8").splitlines()
        collection = init_chroma()
        ids = [f"doc_{i}" for i in range(len(lines))]
        metadatas = [{"source": "user_upload"}] * len(lines)
        collection.add(documents=lines, metadatas=metadatas, ids=ids)
        st.success(f"{len(lines)} documents added to Chroma DB.")

query = st.text_input("💬 Ask a support question:")

if query:
    with st.spinner("Retrieving documents and generating response..."):
        collection = init_chroma()
        results = collection.query(query_texts=[query], n_results=3)
        retrieved_docs = results['documents'][0]
        context = "\n".join(retrieved_docs)

        watson = get_watson_model()
        prompt = f"Context:\n{context}\n\nUser question: {query}\nAnswer:"
        response = watson.generate_text(prompt=prompt)
        st.markdown("### 🤖 SmartResolve says:")
        st.success(response)

        with st.expander("🔍 Retrieved Docs"):
            for doc in retrieved_docs:
                st.markdown(f"- {doc}")
