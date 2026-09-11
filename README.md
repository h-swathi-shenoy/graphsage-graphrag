> GraphSAGE + GraphRAG on Neo4j with AWS Bedrock
>
> A working, runnable example: an LLM-built knowledge graph in Neo4j, a
> GraphSAGE GNN trained on that graph, and a retriever that uses both to answer
> multi-hop questions that plain vector RAG gets wrong.

## What this is

| Layer | Choice |
|---|---|
| Graph DB | Neo4j 5.26 (Docker) + native vector index |
| Extraction + generation | AWS Bedrock, `us.amazon.nova-lite-v1:0` |
| Text embeddings | AWS Bedrock, `amazon.titan-embed-text-v2:0` (1024-d) |
| GNN | GraphSAGE, 2-layer mean aggregator, pure PyTorch (no PyG/DGL) |
| Corpus | 16-doc flight-operations knowledge base for a fictional airline |

## The idea in one paragraph

Vector RAG retrieves chunks that *sound like* the question. It fails when the
answer requires a chain: *Flight 202 departed 3 hours late → Aircraft N882AA
held at the gate → ramp-scheduler ran out of turnaround slots → Staffing
Policy SP-118 mandated extended marshalling on every turnaround → Team
Ground Operations*. No single chunk contains that chain. So we (1) extract an
entity graph with an LLM, (2) train **GraphSAGE** on it so every node's
embedding absorbs its neighbourhood, (3) attach the *question* to the graph as
a virtual node and push it through GraphSAGE to get structure-aware seeds, then
(4) walk 2 hops in Neo4j and feed the LLM both the triples and the chunks.

Proof it learns structure — nearest neighbours of `INC-2209` after training:

```
entity::inc-2209 -> entity::ramp-scheduler (0.63), chunk::inc-2209-rca (0.44), ...
```

`ramp-scheduler` is the true root cause and is **never mentioned** in the
INC-2209 incident document. Pure text embeddings cannot make that link;
GraphSAGE does, because message passing pulls the RCA neighbourhood in.

## Prerequisites

- Docker
- Python 3.10+
- AWS credentials with Bedrock access in `us-east-1` (`aws sts get-caller-identity`)
- Model access enabled for Titan Embed V2 and Nova Lite in the Bedrock console

## Run it

```bash
cd ~/Desktop/VSCodeProjects/graphsage-graphrag

docker compose up -d                     # Neo4j on 7474 (browser) / 7687 (bolt)

python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt

./.venv/bin/python 01_ingest.py           # build the graph   (~1 min, Bedrock calls)
./.venv/bin/python 02_train_graphsage.py  # train the GNN     (~20 s, CPU)
./.venv/bin/python 03_ask.py              # answer the demo questions
```

Or just `./run_all.sh`.

Neo4j browser: <http://localhost:7474> — user `neo4j`, password `graphsage123`.

```cypher
// see the graph
MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 200
// the root-cause chain
MATCH p=(:Entity {key:'flight 202'})-[:RELATES_TO*1..3]-(:Entity {key:'ramp-scheduler'})
RETURN p LIMIT 5
```

## Files

| File | Role |
|---|---|
| `docker-compose.yml` | Neo4j 5.26 + APOC |
| `config.py` | connection settings, model ids, GNN hyperparameters |
| `bedrock.py` | Titan embed / Nova converse wrappers with retry + JSON repair |
| `corpus.py` | the 16-document example knowledge base |
| `01_ingest.py` | LLM entity+relation extraction → Neo4j, embeddings, vector index |
| `graph_store.py` | all Cypher: graph load, vector search, k-hop expansion |
| `sage_model.py` | GraphSAGE layers, mean aggregator, random walks, unsup loss |
| `02_train_graphsage.py` | training loop, writes `e.sage` back to Neo4j |
| `retriever.py` | hybrid retrieval + inductive query embedding + generation |
| `03_ask.py` | CLI: demo set, `--compare`, `--trace` |

## How each step works

### 1. Ingest (`01_ingest.py`)

Each document becomes a `(:Chunk {text, embedding})`. Nova extracts entities and
relationships as strict JSON, which are merged into `(:Entity {key, name, type,
description})` joined by `[:RELATES_TO {type, evidence}]`, plus
`(:Chunk)-[:MENTIONS]->(:Entity)`. `key` is a lowercased name, which is the
(deliberately simple) entity-resolution strategy. Entities get their own Titan
embedding built from name + type + description + the chunks that mention them —
these become the GNN's input features.

Result: 16 chunks, 33 entities, 54 relationships, 70 mentions.

### 2. Train (`02_train_graphsage.py`)

The property graph is flattened into one homogeneous graph (49 nodes, 113
undirected edges): chunks and entities are both nodes, `MENTIONS` and
`RELATES_TO` are both edges. Then standard GraphSAGE:

```
h_v^k = act( W^k · [ h_v^{k-1} ;  mean_{u∈N(v)} h_u^{k-1} ] )
```

implemented as a sparse row-normalised adjacency matmul. Training is
unsupervised (Hamilton et al. eq. 1): positive pairs come from length-4 random
walks, negatives are sampled uniformly.

```
L = -log σ(z_u·z_v · τ) - Σ_neg log σ(-z_u·z_n · τ)
```

Two details that matter, both learned the hard way here:

- **No ReLU on the output layer.** ReLU + L2-normalise on the last layer clips
  every negative coordinate and collapses all embeddings to zero.
- **Temperature `τ`.** Outputs are L2-normalised, so dot products live in
  [-1, 1], the sigmoid saturates and the loss flatlines. Scaling by 10 fixes it.

The 128-d vectors are written back as `Chunk.sage` / `Entity.sage`, and cached
to `graph_cache.pkl` with the model in `sage_model.pt`.

### 3. Retrieve (`retriever.py`)

```
question
  ├─ Titan embed
  ├─ Neo4j vector index  ──────────────► top-4 chunks        (lexical/semantic)
  ├─ virtual node attached to its 5 nearest nodes
  │     └─ GraphSAGE forward pass ─────► top-6 seed entities (structural)
  ├─ Cypher 2-hop expansion from seeds ► ~50 triples
  └─ chunks mentioning the seeds       ► evidence
                                          ↓
                          triples + chunks → Nova Lite → cited answer
```

The virtual-node trick is the point of using GraphSAGE rather than a
transductive method like node2vec: GraphSAGE learns *aggregator functions*, not
a per-node lookup table, so an unseen node — the user's question, or a service
onboarded after training — gets an embedding from its neighbours' features with
**zero retraining**.

### 4. Compare

```bash
./.venv/bin/python 03_ask.py --compare "Which team should be paged for INC-2209 and which team actually caused it?"
```

Vector RAG retrieves the incident docs and names the paged team. GraphRAG
additionally traverses `ramp-scheduler -[OWNED_BY]-> Team Airport Operations`
and `SP-118 -[TRIGGERED_BY]-> Team Ground Operations`, answering both
halves — including facts from `team-groundops`, a document the vector search
never retrieved.

Use `--trace` to print attached nodes, seed entities with SAGE scores, vector
hits, triple count and final context.

## Tuning

| Symptom | Fix |
|---|---|
| Embeddings all identical / zero | Check no ReLU on final layer; lower `SAGE_LR` |
| Loss flatlines near 1.25 | Raise the `scale` temperature in `unsup_loss` |
| Seeds irrelevant | Raise `attach_k`, or `WALK_LENGTH` for wider context |
| Answers miss a hop | `hops=3` in `retrieve()`, raise the `expand()` limit |
| Duplicate entities | Improve `norm()` — alias table or embedding-based resolution |

## Scaling this beyond the demo

- **Batching:** replace full-batch propagation with the paper's neighbour
  sampling (`SAGE_FANOUT` is already in `config.py`) once you exceed ~100k nodes.
- **Neo4j GDS:** for production, `gds.beta.graphSage.train` runs the same
  algorithm inside the database and avoids exporting the graph.
- **Entity resolution** is the real quality lever at scale — lowercase matching
  will not survive a real corpus.
- **Incremental updates:** GraphSAGE is inductive, so new documents only need
  feature extraction plus a forward pass; retrain periodically, not per write.
- **Cost:** ingest is O(docs) Bedrock calls. Cache aggressively; Nova Lite is
  cheap, but extraction on millions of docs is not.
