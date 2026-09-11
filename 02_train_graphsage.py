"""Step 2 - Train GraphSAGE on the Neo4j knowledge graph.

Why GraphSAGE and not just vector search?
  Titan embeddings only know what a node *says*. GraphSAGE learns what a node
  *is connected to* by aggregating neighbour features, so "INC-1043" ends up
  near "settlement-service", "postgres-primary" and "RB-07" even though those
  strings barely overlap textually.

Implementation notes:
  - 2-layer mean-aggregator GraphSAGE (Hamilton et al. 2017), pure PyTorch.
  - Unsupervised loss: random-walk positive pairs + negative sampling.
  - Inductive: forward() takes (features, adjacency) so an unseen query node
    can be embedded at query time without retraining (see graph_store.py).
"""
import pickle

import numpy as np
import torch

import config
from graph_store import GraphStore
from sage_model import (GraphSAGE, build_agg_matrix, random_walk_pairs,
                        unsup_loss)

MODEL_PATH = "sage_model.pt"
GRAPH_PATH = "graph_cache.pkl"


def main():
    rng = np.random.default_rng(42)
    store = GraphStore()

    print("Loading graph from Neo4j...")
    g = store.load_graph()
    n = len(g["node_ids"])
    print(f"  nodes={n}  edges={sum(len(a) for a in g['adj']) // 2}")

    X = torch.tensor(np.array(g["features"]), dtype=torch.float32)
    A = build_agg_matrix(g["adj"], n)
    pairs = random_walk_pairs(g["adj"], n, config.WALK_LENGTH,
                              config.WALKS_PER_NODE, rng)
    print(f"  random-walk positive pairs: {len(pairs)}")

    model = GraphSAGE(X.shape[1], config.SAGE_HIDDEN, config.SAGE_OUT,
                      config.SAGE_LAYERS)
    opt = torch.optim.Adam(model.parameters(), lr=config.SAGE_LR)

    print("Training GraphSAGE...")
    for epoch in range(1, config.SAGE_EPOCHS + 1):
        model.train()
        opt.zero_grad()
        Z = model(X, A)
        batch = pairs[rng.choice(len(pairs), size=min(2048, len(pairs)),
                                 replace=False)]
        loss = unsup_loss(Z, batch, n, rng, config.NEG_SAMPLES)
        loss.backward()
        opt.step()
        if epoch % 25 == 0 or epoch == 1:
            print(f"  epoch {epoch:4d}  loss {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        Z = model(X, A).numpy()

    print("Writing GraphSAGE embeddings back to Neo4j...")
    store.save_sage_embeddings(g["node_ids"], g["node_labels"], Z)

    torch.save({"state_dict": model.state_dict(),
                "in_dim": X.shape[1],
                "hidden": config.SAGE_HIDDEN,
                "out": config.SAGE_OUT,
                "layers": config.SAGE_LAYERS}, MODEL_PATH)
    with open(GRAPH_PATH, "wb") as f:
        pickle.dump({"node_ids": g["node_ids"], "node_labels": g["node_labels"],
                     "adj": g["adj"], "features": np.array(g["features"]),
                     "sage": Z}, f)
    print(f"Saved {MODEL_PATH} and {GRAPH_PATH}")

    # quick sanity check: nearest neighbours in GraphSAGE space
    print("\nGraphSAGE nearest neighbours (structure-aware):")
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    for probe in ["entity::inc-2209", "entity::flight 202",
                  "chunk::inc-2209"]:
        if probe not in g["node_ids"]:
            continue
        i = g["node_ids"].index(probe)
        sims = Zn @ Zn[i]
        top = np.argsort(-sims)[1:6]
        print(f"  {probe:20s} -> " +
              ", ".join(f"{g['node_ids'][j]}({sims[j]:.2f})" for j in top))

    store.close()


if __name__ == "__main__":
    main()
