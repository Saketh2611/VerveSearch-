# 🧠 Mental Health Therapy Semantic Search

Live Demo: [Click here to try it](https://mentalhealth2611.streamlit.app/)

This project is a natural language search engine that helps retrieve the most relevant mental health therapy responses based on user input. Instead of relying on keywords, it uses semantic embeddings to understand the meaning behind a user's concern and return relevant counseling advice.

---

## 📌 Features

- 🔍 **Semantic Search:** Powered by [Sentence Transformers](https://www.sbert.net/) and [Qdrant Vector DB](https://qdrant.tech/)
- 🎯 **Real Therapist Responses:** Searches actual mental health counseling conversations
- 📏 **Similarity Score Filter:** View only highly relevant results
- 🎛️ **Keyword Filters:** Optional filters like `sports`, `sleep`, `anxiety`, `relationships`, etc.
- ⚡ **Instant Results:** Deployed live using Streamlit Cloud

---

## 💡 Example Queries

- "I feel anxious before going to school"
- "How do I handle social pressure?"
- "I'm not sleeping well lately"
- "I feel burned out from sports"

---

## 🛠️ Tech Stack

| Component | Details |
|----------|---------|
| 🧠 Embedding Model | `paraphrase-mpnet-base-v2` from HuggingFace |
| 📦 Vector DB | Qdrant Cloud (COSINE distance) |
| 🧱 Backend | Python + Sentence Transformers |
| 🌐 UI | Streamlit |
| ☁️ Hosting | Streamlit Community Cloud |

---

## 📁 Folder Structure

