"""GraphRAG retriever: Titan vector search + inductive GraphSAGE + graph walk.

Retrieval pipeline for a question q:

  1. Titan-embed q.
  2. VECTOR SEEDS   - Neo4j vector index -> top-k lexically/semantically similar
                      chunks.
  3. SAGE SEEDS     - q is attached to the graph as a *virtual node* whose
                      neighbours are its nearest existing nodes in raw Titan
                      space, then pushed through the trained GraphSAGE layers.
                      This is GraphSAGE's inductive property: an unseen node
                      gets a structure-aware embedding with zero retraining.
                      Nearest entities in SAGE space become graph seeds.
  4. GRAPH EXPANSION- walk 1-2 hops from the seed entities in Neo4j and collect
                      the triples plus every chunk that mentions them.
  5. CONTEXT        - triples (the reasoning skeleton) + chunk text (evidence)
                      are handed to Bedrock Nova.
"""
import pickle

import numpy as np
import torch

import bedrock
from graph_store import ENTITY_P, GraphStore
from sage_model import GraphSAGE, build_agg_matrix


class GraphRAG:
    def __init__(self, model_path="sage_model.pt", graph_path="graph_cache.pkl"):
        ckpt = torch.load(model_path, map_location="cpu")
        self.model = GraphSAGE(ckpt["in_dim"], ckpt["hidden"],
                               ckpt["out"], ckpt["layers"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

        with open(graph_path, "rb") as f:
            g = pickle.load(f)
        self.node_ids = g["node_ids"]
        self.node_labels = g["node_labels"]
        self.adj = g["adj"]
        self.X = g["features"].astype(np.float32)
        self.Z = g["sage"].astype(np.float32)
        self.Zn = self.Z / (np.linalg.norm(self.Z, axis=1, keepdims=True) + 1e-9)
        self.Xn = self.X / (np.linalg.norm(self.X, axis=1, keepdims=True) + 1e-9)
        self.store = GraphStore()

    def close(self):
        self.store.close()

    # ---------- inductive query embedding ----------

    def sage_embed_query(self, q_emb, attach_k=5):
        """Embed an unseen query node by attaching it to its nearest nodes."""
        q = np.asarray(q_emb, dtype=np.float32)
        qn = q / (np.linalg.norm(q) + 1e-9)
        nbrs = np.argsort(-(self.Xn @ qn))[:attach_k].tolist()

        n = len(self.node_ids)
        X_ext = np.vstack([self.X, q[None, :]])
        adj_ext = [list(a) for a in self.adj] + [nbrs]   # directed: query -> nbrs

        A = build_agg_matrix(adj_ext, n + 1)
        with torch.no_grad():
            Z = self.model(torch.tensor(X_ext), A).numpy()
        return Z[n], nbrs

    # ---------- retrieval ----------

    def retrieve(self, question, vec_k=4, sage_k=6, hops=2):
        q_emb = bedrock.embed(question)
        vector_hits = self.store.vector_search_chunks(q_emb, k=vec_k)

        z_q, attached = self.sage_embed_query(q_emb)
        zqn = z_q / (np.linalg.norm(z_q) + 1e-9)
        sims = self.Zn @ zqn

        ent_idx = [i for i, l in enumerate(self.node_labels) if l == "Entity"]
        ranked = sorted(ent_idx, key=lambda i: -sims[i])[:sage_k]
        seed_keys = [self.node_ids[i][len(ENTITY_P):] for i in ranked]
        seed_scores = {self.node_ids[i][len(ENTITY_P):]: float(sims[i])
                       for i in ranked}

        # entities mentioned by the vector-hit chunks also make good seeds
        triples = self.store.expand(seed_keys, hops=hops)
        graph_chunks = self.store.chunks_for_entities(seed_keys, limit=8)

        chunks = {c["id"]: c["text"] for c in vector_hits}
        for c in graph_chunks:
            chunks.setdefault(c["id"], c["text"])

        return {
            "question": question,
            "vector_hits": vector_hits,
            "attached_nodes": [self.node_ids[i] for i in attached],
            "seed_entities": seed_scores,
            "triples": triples,
            "chunks": chunks,
            "entities": self.store.entity_meta(seed_keys),
        }

    # ---------- generation ----------

    @staticmethod
    def build_context(r):
        lines = ["## Key entities"]
        for e in r["entities"]:
            lines.append(f"- {e['name']} [{e['type']}]: {e['description']}")
        lines.append("\n## Graph relationships (source -[rel]-> target)")
        for t in r["triples"]:
            ev = f"  // {t['evidence']}" if t["evidence"] else ""
            lines.append(f"- {t['source']} -[{t['rel']}]-> {t['target']}{ev}")
        lines.append("\n## Source documents")
        for cid, text in r["chunks"].items():
            lines.append(f"[{cid}] {text}")
        return "\n".join(lines)

    def answer(self, question, **kw):
        r = self.retrieve(question, **kw)
        ctx = self.build_context(r)
        prompt = f"""Answer the question using ONLY the context below. The graph
relationships show how entities connect - use them to reason across documents.
Cite source document ids in square brackets. If the context is insufficient,
say so.

CONTEXT:
{ctx}

QUESTION: {question}

ANSWER:"""
        return self.answer_text(prompt), r

    @staticmethod
    def answer_text(prompt):
        return bedrock.chat(prompt, system="You are a precise SRE assistant.")

    # ---------- baseline for comparison ----------

    def baseline_answer(self, question, k=4):
        q_emb = bedrock.embed(question)
        hits = self.store.vector_search_chunks(q_emb, k=k)
        ctx = "\n".join(f"[{h['id']}] {h['text']}" for h in hits)
        prompt = f"""Answer the question using ONLY the context below. Cite source
document ids in square brackets. If the context is insufficient, say so.

CONTEXT:
{ctx}

QUESTION: {question}

ANSWER:"""
        return self.answer_text(prompt), hits
