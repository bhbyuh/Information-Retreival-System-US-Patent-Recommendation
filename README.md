# US Patent Recommendation System

This repository contains an end-to-end semantic information retrieval system for recommending US patents. It uses semantic similarity and reranking techniques to return the most relevant patents to a user's query. The system is built with a Jupyter Notebook for data processing and a FastAPI backend for serving recommendations.

---

## 🔍 Features

- Load and preprocess US patent data from CSV
- Embed patent descriptions and store them in a Vector Database (VDB)
- Semantic search using vector similarity
- Rerank top results using a reranker model
- Expose a FastAPI endpoint for querying and retrieving results
