import streamlit as st
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance
import os

# -----------------------------
# 🔧 App Configuration
# -----------------------------
st.set_page_config(page_title="🧠 Therapy Search Engine", layout="wide")
st.title("🧠 Mental Health Therapy Semantic Search")
st.markdown("Use natural language to find relevant therapy responses. Powered by Sentence Transformers + Qdrant.")

# -----------------------------
# 🧠 Load the Embedding Model
# -----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("paraphrase-mpnet-base-v2")

model = load_model()

# -----------------------------
# 🔌 Connect to Qdrant
# -----------------------------
QDRANT_URL = "https://23a37241-1707-4f1a-8f5e-47c00502551d.us-west-1-0.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY   = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.6MHdGWXVS2dEszyAaokzSlQbqe0Fdh_vFEvBJxXH50c"  # 👈  Replace this with your actual API key

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

COLLECTION_NAME = "therapy_logs"

# -----------------------------
# 🔍 UI Input Components
# -----------------------------
st.subheader("🔍 Search Your Concern")

query = st.text_input("💬 What mental health concern are you facing?", placeholder="e.g., I feel anxious all the time")

top_k = st.slider("📊 Number of results to retrieve", min_value=1, max_value=10, value=5)

min_score = st.slider("📏 Minimum similarity score", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

keyword_filter = st.text_input("🔤 Filter results by keyword (optional)", placeholder="e.g., anxiety, school, pressure,sports")

# -----------------------------
# 🚀 Perform Search
# -----------------------------
if st.button("Search") and query.strip():
    with st.spinner("Embedding your query and searching Qdrant..."):
        query_vector = model.encode(query).tolist()

        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k
        )

        # ✅ Filter results by score + optional keyword
        filtered_results = [
            r for r in results
            if r.score >= min_score and
               (keyword_filter.lower() in r.payload['context'].lower() or keyword_filter == "")
        ]

    if filtered_results:
        st.success(f"🎯 Showing {len(filtered_results)} filtered results for: *{query}*")
        for i, result in enumerate(filtered_results, 1):
            st.markdown(f"### 🧠 Result {i}")
            st.markdown(f"**Context:** {result.payload.get('context', '-')}")
            st.markdown(f"**Response:** {result.payload.get('response', '-')}")
            st.markdown(f"**Similarity Score:** `{result.score:.4f}`")
            st.markdown("---")
    else:
        st.warning("No results found with the selected filters.")

elif query.strip() == "":
    st.info("Enter a query and click 'Search' to begin.")

