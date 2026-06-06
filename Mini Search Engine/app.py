import streamlit as st
import os
from utils.pdf_processor import extract_text_from_pdf, split_into_chunks
from utils.embeddings import get_embedding
from utils.pinecone_client import init_pinecone, upsert_chunks, query_index

st.set_page_config(
    page_title="Mini Search Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

/* Dark background */
.stApp {
    background: #0d0f14;
    color: #e8eaf0;
}

/* Header */
.hero-title {
    font-size: 3.2rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #e8eaf0 0%, #7b88ff 60%, #ff6b9d 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.hero-sub {
    font-size: 1rem;
    color: #6b7280;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.05em;
    margin-bottom: 2rem;
}

/* Cards */
.result-card {
    background: #161a24;
    border: 1px solid #252a38;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s ease;
}
.result-card:hover {
    border-color: #7b88ff55;
}
.result-doc {
    font-size: 0.75rem;
    font-family: 'DM Mono', monospace;
    color: #7b88ff;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.result-score {
    font-size: 0.7rem;
    font-family: 'DM Mono', monospace;
    color: #ff6b9d;
    float: right;
}
.result-text {
    font-size: 0.92rem;
    color: #c8cad4;
    line-height: 1.7;
    margin-top: 0.6rem;
}

/* Score bar */
.score-bar-wrap {
    background: #252a38;
    border-radius: 4px;
    height: 4px;
    margin-top: 0.8rem;
}
.score-bar {
    height: 4px;
    border-radius: 4px;
    background: linear-gradient(90deg, #7b88ff, #ff6b9d);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d0f14;
    border-right: 1px solid #1e2230;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #7b88ff, #ff6b9d) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
    letter-spacing: 0.02em !important;
}
.stButton > button:hover {
    opacity: 0.9 !important;
}

/* Text input */
.stTextInput > div > div > input {
    background: #161a24 !important;
    border: 1px solid #252a38 !important;
    border-radius: 8px !important;
    color: #e8eaf0 !important;
    font-family: 'Syne', sans-serif !important;
}

/* File uploader */
.stFileUploader {
    border: 1.5px dashed #252a38 !important;
    border-radius: 10px !important;
    background: #161a24 !important;
}

/* Divider */
hr { border-color: #1e2230 !important; }

/* Status tags */
.tag-indexed {
    display: inline-block;
    background: #1a2e1a;
    color: #4ade80;
    border: 1px solid #2a4a2a;
    border-radius: 20px;
    font-size: 0.68rem;
    font-family: 'DM Mono', monospace;
    padding: 2px 10px;
    margin-left: 8px;
}
</style>
""", unsafe_allow_html=True)


if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []
if "index" not in st.session_state:
    st.session_state.index = None


with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")

    pinecone_api_key = st.text_input("Pinecone API Key", type="password", placeholder="pcsk_...")
    pinecone_index   = st.text_input("Index Name", value="mini-search", placeholder="my-index")
    openai_api_key   = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")

    st.markdown("---")
    st.markdown("**Settings**")
    chunk_size    = st.slider("Chunk size (chars)", 200, 1000, 500, 50)
    chunk_overlap = st.slider("Chunk overlap (chars)", 0, 200, 50, 10)
    top_k         = st.slider("Top-K results", 1, 10, 5)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem; color:#4b5563; font-family: DM Mono, monospace; line-height:1.6;'>
    📌 Needs Pinecone + OpenAI keys<br>
    📌 Dimension: 1536 (text-embedding-3-small)<br>
    📌 Metric: cosine
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="hero-title">🔍 Mini Search Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">semantic · vector · pdf</div>', unsafe_allow_html=True)

tab_upload, tab_search = st.tabs(["📄 Upload & Index", "🔎 Search"])


with tab_upload:
    st.markdown("#### Upload PDF Documents")
    st.caption("Upload at least 5 PDFs. Text will be extracted, chunked, embedded, and stored in Pinecone.")

    uploaded_files = st.file_uploader(
        "Drop your PDFs here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} file(s) selected:**")
        for f in uploaded_files:
            already = f.name in st.session_state.indexed_files
            tag = '<span class="tag-indexed">indexed</span>' if already else ""
            st.markdown(f"&nbsp;&nbsp;• `{f.name}` {tag}", unsafe_allow_html=True)

        st.markdown("")
        if st.button("⚡ Index All Documents"):
            if not pinecone_api_key or not openai_api_key:
                st.error("Please enter your Pinecone and OpenAI API keys in the sidebar.")
            elif len(uploaded_files) < 5:
                st.warning("Please upload at least 5 PDF files.")
            else:
                os.environ["OPENAI_API_KEY"]   = openai_api_key
                os.environ["PINECONE_API_KEY"] = pinecone_api_key

                with st.spinner("Initializing Pinecone index…"):
                    try:
                        index = init_pinecone(pinecone_api_key, pinecone_index)
                        st.session_state.index = index
                    except Exception as e:
                        st.error(f"Pinecone init failed: {e}")
                        st.stop()

                progress = st.progress(0, text="Starting…")
                total    = len(uploaded_files)

                for i, pdf_file in enumerate(uploaded_files):
                    progress.progress((i) / total, text=f"Processing {pdf_file.name}…")

                    try:
                        text   = extract_text_from_pdf(pdf_file)
                        chunks = split_into_chunks(text, chunk_size, chunk_overlap)
                        upsert_chunks(index, chunks, pdf_file.name, openai_api_key)

                        if pdf_file.name not in st.session_state.indexed_files:
                            st.session_state.indexed_files.append(pdf_file.name)

                        st.success(f"✅ {pdf_file.name} — {len(chunks)} chunks indexed")
                    except Exception as e:
                        st.error(f"❌ {pdf_file.name}: {e}")

                progress.progress(1.0, text="Done!")
                st.balloons()

    if st.session_state.indexed_files:
        st.markdown("---")
        st.markdown(f"**Indexed documents ({len(st.session_state.indexed_files)}):**")
        for name in st.session_state.indexed_files:
            st.markdown(f"&nbsp;&nbsp;✔ `{name}`", unsafe_allow_html=True)


with tab_search:
    st.markdown("#### Semantic Search")
    st.caption("Ask anything. The engine retrieves the most relevant passages from your indexed PDFs.")

    col_q, col_btn = st.columns([5, 1])
    with col_q:
        query = st.text_input("Your query", placeholder="e.g. What is vector search?", label_visibility="collapsed")
    with col_btn:
        search_clicked = st.button("Search", use_container_width=True)

    if search_clicked:
        if not query.strip():
            st.warning("Please enter a search query.")
        elif not pinecone_api_key or not openai_api_key:
            st.error("Please enter your API keys in the sidebar first.")
        else:
            os.environ["OPENAI_API_KEY"]   = openai_api_key
            os.environ["PINECONE_API_KEY"] = pinecone_api_key

            with st.spinner("Searching…"):
                try:
                    if st.session_state.index is None:
                        st.session_state.index = init_pinecone(pinecone_api_key, pinecone_index)

                    results = query_index(
                        st.session_state.index,
                        query,
                        openai_api_key,
                        top_k=top_k,
                    )

                    if not results:
                        st.info("No results found. Make sure you've indexed some documents first.")
                    else:
                        st.markdown(f"**{len(results)} result(s) for:** _{query}_")
                        st.markdown("")
                        for match in results:
                            score   = match["score"]
                            doc     = match.get("metadata", {}).get("source", "Unknown")
                            chunk_i = match.get("metadata", {}).get("chunk_index", "?")
                            text    = match.get("metadata", {}).get("text", "")

                            bar_pct = int(score * 100)
                            st.markdown(f"""
                            <div class="result-card">
                                <div>
                                    <span class="result-doc">📄 {doc} &nbsp;·&nbsp; chunk #{chunk_i}</span>
                                    <span class="result-score">score {score:.4f}</span>
                                </div>
                                <div class="result-text">{text}</div>
                                <div class="score-bar-wrap">
                                    <div class="score-bar" style="width:{bar_pct}%;"></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Search failed: {e}")
