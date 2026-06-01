import os
import time
import faiss
import numpy as np
import streamlit as st

import config
from utils import Utils
from dataset import Dataset
from embeddings import Embeddings
from retriever import Retriever
from reranker import Reranker
from llm import LLM

st.set_page_config(
    page_title="RAG Pipeline",
    layout="wide",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #888;
        margin-bottom: 2rem;
    }
    .doc-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_documents():
    dataset = Dataset(config.DATASET_NAME, config.DATASET_SPLIT)
    return dataset.get_dataset()


@st.cache_resource
def load_embed_model():
    return Embeddings(config.EMBEDDING_MODEL_NAME)


@st.cache_resource
def load_or_build_index(_documents, _embed_model):
    Utils.check_dir(config.INDEX_DIR)

    if os.path.exists(config.INDEX_PATH):
        return faiss.read_index(config.INDEX_PATH)

    index = faiss.IndexFlatL2(config.EMBEDDING_DIMENSION)
    for i in range(0, len(_documents), config.BATCH_SIZE):
        batch = _documents[i : i + config.BATCH_SIZE]
        embeds = _embed_model.get_embedding(batch["text"])
        index.add(np.array(embeds))

    faiss.write_index(index, config.INDEX_PATH)
    return index


@st.cache_resource
def load_reranker():
    return Reranker(config.RERANKER_MODEL_NAME)


st.markdown('<p class="main-header">Advanced RAG Pipeline</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">'
    'Semantic search and question answering over arXiv papers.'
    '</p>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Retriever Top K", 5, 50, config.RETRIEVER_TOP_K)
    top_n = st.slider("Reranker Top N", 1, 20, config.RERANKER_TOP_N)
    show_sources = st.checkbox("Show source documents", value=True)

query = st.text_input(
    "Ask a question:",
    placeholder="e.g., Can you explain why we would want to do RLHF?",
)

if st.button("Search & Generate", type="primary", use_container_width=True) and query:
    documents = load_documents()
    embed_model = load_embed_model()
    index = load_or_build_index(documents, embed_model)
    reranker = load_reranker()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Retrieved Documents")

        start = time.time()
        docs = Retriever.search(
            documents=documents,
            embed_model=embed_model,
            index=index,
            query=query,
            top_k=top_k,
        )
        retrieval_time = time.time() - start

        start = time.time()
        reranked = reranker.rerank(docs, query, top_n=top_n)
        rerank_time = time.time() - start

        st.caption(
            f"Retrieval: {retrieval_time:.2f}s  |  Reranking: {rerank_time:.2f}s"
        )

        if show_sources:
            for i, (doc_text, score) in enumerate(reranked, 1):
                score_val = float(score)
                with st.expander(f"Document {i} (Similarity: {score_val:.4f})"):
                    st.markdown(doc_text[:1000] + ("..." if len(doc_text) > 1000 else ""))

    with col2:
        st.subheader("Answer")
        context = "\n".join([doc[0] for doc in reranked])

        try:
            start = time.time()
            llm = LLM(model=config.LLM_MODEL_NAME, temperature=config.LLM_TEMPERATURE)
            answer = llm.generate(query=query, context=context)
            llm_time = time.time() - start
            st.caption(f"LLM Generation: {llm_time:.2f}s")
            st.markdown(answer)
        except Exception as e:
            st.error(f"Error: {e}")
