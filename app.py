import streamlit as st
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance
import os

# -----------------------------
# 🔧 App Config
# -----------------------------
st.set_page_config(page_title="🧠 Therapy Search Engine", layout="wide")
st.title("🧠 Mental Health Therapy Semantic Search")
st.markdown("Search real counseling logs by meaning, not keywords.")

# -----------------------------
# 🧠 Load Embedding Model
# -----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("paraphrase-mpnet-base-v2")

model = load_model()

# -----------------------------
# 🔌 Connect to Qdrant Cloud
# -----------------------------
QDRANT_URL = "https://23a37241-1707-4f1a-8f5e-47c00502551d.us-west-1-0.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.6MHdGWXVS2dEszyAaokzSlQbqe0Fdh_vFEvBJxXH50c"  # 👈 Replace with your actual API key

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

COLLECTION_NAME = "therapy_logs"

# -----------------------------
# 🔍 Search UI
# -----------------------------
query = st.text_input("💬 What mental health concern are you facing?", placeholder="e.g., I feel overwhelmed with my emotions")

top_k = st.slider("🔢 Number of responses to show", 1, 10, 5)

if st.button("Search") and query.strip():
    with st.spinner("Embedding and searching..."):
        query_vector = model.encode(query).tolist()

        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k
        )

    st.success(f"Top {top_k} results for: *{query}*")
    for i, result in enumerate(results, 1):
        st.markdown(f"### 🎯 Result {i}")
        st.markdown(f"**🧠 Context:** {result.payload.get('context', '—')}")
        st.markdown(f"**💬 Therapist Response:** {result.payload.get('response', '—')}")
        st.markdown(f"**🔗 Similarity Score:** `{result.score:.4f}`")
        st.markdown("---")

elif query.strip() == "":
    st.info("Please enter a query to begin searching.")
