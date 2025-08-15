import os
import time
import streamlit as st
from modules.utils import simple_chunk_text, parse_docs
from modules.vector_store import VectorStore
from modules.graph_store import GraphStore
from modules.hybrid_retrieval import hybrid_retrieve, extract_simple_entities_spacy

import spacy
from pyvis.network import Network
from openai import OpenAI

st.set_page_config(page_title="Hybrid RAG (Vector + Graph)", page_icon="🧠", layout="wide")

st.sidebar.title("⚙️ Settings")
persist_dir = st.sidebar.text_input("Vector DB directory", value="storage")
collection_name = st.sidebar.text_input("Collection name", value="edu_docs")
embedding_model = st.sidebar.text_input("Embedding model (SentenceTransformers)", value="all-MiniLM-L6-v2")
top_k = st.sidebar.slider("Top-K (vector)", min_value=3, max_value=15, value=5, step=1)
graph_depth = st.sidebar.slider("Graph traversal depth", min_value=0, max_value=3, value=1, step=1)
alpha = st.sidebar.slider("Merge weight α (vector)", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
show_graph = st.sidebar.checkbox("Show knowledge graph (pyvis)", value=True)
regenerate = st.sidebar.button("🔁 Rebuild Index (from data/sample_docs.txt)")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    st.warning("🔑 Set OPENAI_API_KEY in your environment or Streamlit secrets to enable answer generation.")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

@st.cache_resource(show_spinner=False)
def load_spacy_model():
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

vs = VectorStore(persist_dir=persist_dir, collection_name=collection_name, model_name=embedding_model)
gs = GraphStore()

def build_index():
    with open("data/sample_docs.txt", "r", encoding="utf-8") as f:
        raw = f.read()
    docs = parse_docs(raw)

    vs.reset()

    ids, texts, metas = [], [], []
    for i, d in enumerate(docs):
        chunks = simple_chunk_text(d["content"], max_chars=800, overlap=80)
        for j, ch in enumerate(chunks):
            cid = f"doc{i}_chunk{j}"
            ids.append(cid)
            texts.append(ch)
            metas.append({"id": cid, "title": d["title"], "source": "sample_docs.txt"})

    vs.add(ids=ids, texts=texts, metadatas=metas)

    for cid, text in zip(ids, texts):
        ents = extract_simple_entities_spacy(nlp, text)
        for e in ents:
            gs.link_entity_to_chunk(e, cid)
        for a_i in range(len(ents)):
            for b_i in range(a_i + 1, len(ents)):
                gs.add_relation(ents[a_i], ents[b_i], rel="co_occurs")
    return len(ids)

if regenerate or vs.collection.count() == 0:
    with st.spinner("Building index & knowledge graph..."):
        n_chunks = build_index()
    st.success(f"Indexed {n_chunks} chunks and constructed knowledge graph.")

st.title("🧠 Hybrid RAG with Graph Knowledge Integration")
st.caption("Vector search (ChromaDB) + Knowledge Graph (NetworkX) + OpenAI generation")

query = st.text_input("Ask a question about the education AI dataset:", value="How can AI support students with disabilities?")

col1, col2 = st.columns([1.2, 0.8])

with col1:
    if st.button("🔎 Retrieve & Generate", type="primary"):
        t0 = time.time()
        results = hybrid_retrieve(vs, gs, nlp, query_text=query, k_vec=top_k, graph_depth=graph_depth, alpha=alpha)
        t1 = time.time()

        st.subheader("Merged Retrieval Results")
        st.write(f"Latency: {(t1 - t0):.3f}s")
        st.json({
            "graph_entities": results["graph_entities"],
            "top_merged_ids": [m["id"] for m in results["merged_chunks"][:5]]
        })

        context_blocks = []
        taken = set()
        for item in results["merged_chunks"][:8]:
            if item["doc"] and item["id"] not in taken:
                context_blocks.append(f"[{item['meta'].get('title','')}] {item['doc']}")
                taken.add(item["id"])

        final_answer = "⚠️ Set OPENAI_API_KEY to enable LLM answer."
        if client and context_blocks:
            prompt = f'''You are a helpful educational AI assistant.
Use the provided context to answer the user's query concisely and with clear structure.
Cite section titles in square brackets like [Title] when relevant.

Query: {query}

Context:
{chr(10).join('- ' + c for c in context_blocks)}

Answer:'''
            try:
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant for education-related questions."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.2,
                    )
                    final_answer = resp.choices[0].message.content.strip()
            except Exception as e:
                    final_answer = f"OpenAI API error: {e}"

        st.markdown("### 🧩 Answer")
        st.write(final_answer)

        st.markdown("---")
        st.markdown("### 🔍 Top Vector Results")
        for r in results["vector_results"]:
            with st.expander(f"{r['id']} (sim={r['similarity']}) — {r['meta'].get('title','')}"):
                st.write(r["doc"])

with col2:
        st.subheader("Knowledge Graph")
        if show_graph:
            net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="#222222", notebook=False, directed=False)
            nodes, edges = gs.to_pyvis()
            for n in nodes:
                net.add_node(n["id"], label=n["label"], title=n["title"])
            for e in edges:
                net.add_edge(e["from"], e["to"], title=e["title"], value=e["value"])
            html_path = "graph.html"
            net.show(html_path)
            with open(html_path, "r", encoding="utf-8") as f:
                html = f.read()
            st.components.v1.html(html, height=620, scrolling=True)
        else:
            st.info("Enable 'Show knowledge graph' in the sidebar to visualize entity relations.")

st.markdown("---")
st.markdown("#### How to use")
st.markdown("""
    1. Put your domain data into `data/your_docs.txt` with blocks like:

       ```
       Title: ...
       Content: ...
       ```

    2. Click **Rebuild Index** to re-embed and reconstruct the knowledge graph.
    3. Ask a question and click **Retrieve & Generate**.
""")
