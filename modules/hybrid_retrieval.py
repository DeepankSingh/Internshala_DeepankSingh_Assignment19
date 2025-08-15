from typing import Dict, Any, List
import re

def extract_simple_entities_spacy(nlp, text: str) -> List[str]:
    doc = nlp(text)
    ents = []
    for ent in doc.ents:
        if ent.label_ in {"PERSON","ORG","GPE","NORP","EVENT","WORK_OF_ART","LAW","PRODUCT"}:
            ents.append(ent.text.strip())
    for token in doc:
        if token.pos_ in {"NOUN","PROPN"} and len(token.text) > 2:
            ents.append(token.lemma_.lower())
    seen = set()
    uniq = []
    for e in ents:
        if e not in seen:
            seen.add(e)
            uniq.append(e)
    return uniq[:25]

def hybrid_retrieve(vs, gs, nlp, query_text: str, k_vec: int = 5, graph_depth: int = 1, alpha: float = 0.6) -> Dict[str, Any]:
    vres = vs.query(query_text, k=k_vec)
    vec_ids = vres.get("ids", [[]])[0] if "ids" in vres else []
    vec_docs = vres.get("documents", [[]])[0] if "documents" in vres else []
    vec_metas = vres.get("metadatas", [[]])[0] if "metadatas" in vres else []
    vec_dists = vres.get("distances", [[]])[0] if "distances" in vres else []
    vec_sims = [1.0 - d for d in vec_dists]

    q_entities = extract_simple_entities_spacy(nlp, query_text)
    expanded = gs.expand_from_query_entities(q_entities, depth=graph_depth, top_entities=20)
    graph_chunk_ids = gs.chunks_from_entities(expanded, max_chunks=50)

    score_map = {}
    for cid, sim in zip(vec_ids, vec_sims):
        score_map[cid] = score_map.get(cid, 0.0) + alpha * sim

    if graph_chunk_ids:
        g_increment = (1.0 - alpha)
        for cid in graph_chunk_ids:
            score_map[cid] = score_map.get(cid, 0.0) + g_increment

    merged_rank = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
    id_to_docmeta = {m.get("id"): (d, m) for d, m in zip(vec_docs, vec_metas) if m and "id" in m}

    merged = []
    for cid, score in merged_rank:
        doc, meta = id_to_docmeta.get(cid, (None, None))
        merged.append({"id": cid, "score": round(score, 4), "doc": doc, "meta": meta})

    return {
        "vector_results": [{
            "id": m.get("id"),
            "similarity": round(s, 4),
            "doc": d,
            "meta": m
        } for d, m, s in zip(vec_docs, vec_metas, vec_sims)],
        "graph_entities": expanded,
        "merged_chunks": merged
    }
