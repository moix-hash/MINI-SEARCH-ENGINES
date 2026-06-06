#  Mini Search Engine — Semantic PDF Search

A Google-like semantic search app built with **Streamlit**, **OpenAI Embeddings**, and **Pinecone**.  
Upload PDFs → extract & chunk text → embed → store in Pinecone → search semantically.

---

##  Project Structure

```
mini_search_engine/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── utils/
    ├── __init__.py
    ├── pdf_processor.py    # PDF text extraction + chunking
    ├── embeddings.py       # OpenAI embedding calls
    └── pinecone_client.py  # Pinecone init, upsert, query
```

---

##  Setup

### 1. Clone / download

```bash
git clone <your-repo-url>
cd mini_search_engine
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. API Keys

You need two API keys:

| Service   | Where to get it                              |
|-----------|----------------------------------------------|
| **OpenAI**    | https://platform.openai.com/api-keys        |
| **Pinecone**  | https://app.pinecone.io → API Keys          |

Enter them in the **sidebar** of the app at runtime (no `.env` required).

---

##  Run locally

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

---

##  How to Use

### Step 1 — Upload & Index
1. Go to the **"Upload & Index"** tab.
2. Upload **at least 5 PDF files**.
3. Enter your API keys in the sidebar.
4. Click **"⚡ Index All Documents"**.  
   Each PDF will be: extracted → chunked → embedded → stored in Pinecone.

### Step 2 — Search
1. Go to the **"Search"** tab.
2. Type a natural-language query (e.g. _"What is vector search?"_).
3. Click **"Search"**.  
   The app returns the **Top-K** most relevant chunks with:
   -  Source document name
   -  Similarity score (cosine)
   -  Retrieved paragraph

---

##  Deploy to Streamlit Cloud

1. Push code to a **public GitHub repo**.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select your repo and set **Main file path** to `app.py`.
4. Click **Deploy** — no secrets needed (keys are entered in the UI).

---

##  Configuration (Sidebar)

| Option           | Default             | Description                               |
|------------------|---------------------|-------------------------------------------|
| Pinecone API Key | —                   | Your Pinecone key                         |
| Index Name       | `mini-search`       | Pinecone index (auto-created if missing)  |
| OpenAI API Key   | —                   | Your OpenAI key                           |
| Chunk size       | 500 chars           | Max characters per chunk                  |
| Chunk overlap    | 50 chars            | Overlap between adjacent chunks           |
| Top-K results    | 5                   | Number of results to return               |

---

##  Technical Details

- **Embedding model**: `text-embedding-3-small` (OpenAI) — dimension: **1536**
- **Vector DB**: Pinecone Serverless (AWS us-east-1)
- **Similarity metric**: Cosine similarity
- **Chunking**: Character-level sliding window with configurable overlap
- **PDF parsing**: `pypdf` library

---

## Dependencies

```
streamlit>=1.35.0
pypdf>=4.0.0
openai>=1.30.0
pinecone>=4.1.0
```

---

##  Learning Outcomes

-  Understand how **text embeddings** represent semantic meaning
-  Use **Pinecone** as a vector database for similarity search
-  Implement **semantic / retrieval-augmented search**
-  Build a **full-stack AI app** with Streamlit
