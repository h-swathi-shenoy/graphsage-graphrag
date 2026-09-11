"""Step 1 - Build the knowledge graph in Neo4j.

For every document we:
  1. create a (:Chunk) node holding the raw text + a Titan embedding,
  2. ask Nova (Bedrock) to extract entities and relationships as JSON,
  3. merge those into (:Entity) nodes joined by :RELATES_TO edges,
  4. link chunk -> entity with :MENTIONS.

The resulting property graph is what both the Cypher retriever and the
GraphSAGE trainer consume.
"""
import json

from neo4j import GraphDatabase

import bedrock
import config
from corpus import get_docs

EXTRACTION_SYSTEM = """You are an information-extraction engine for an
airline flight-operations knowledge graph. Return STRICT JSON
only, no prose, no markdown."""

EXTRACTION_PROMPT = """Extract entities and relationships from the text.

Allowed entity types: Flight, Equipment, System, Team, Person, Incident,
Runbook, Policy, StaffingPolicy, Technology, Metric.

Rules:
- "name" must be the exact identifier used in the text (e.g. "Flight 202",
  "INC-2209", "Team Ground Operations", "ramp-scheduler", "SP-118",
  "Aircraft N882AA").
- Relationship "type" must be an UPPER_SNAKE_CASE verb phrase such as
  CALLS, DEPENDS_ON, OWNED_BY, OWNS, LED_BY, CAUSED_BY, TRIGGERED_BY,
  RESOLVED_BY, AFFECTS, CONNECTS_TO, VIOLATES, DEFINES, PART_OF.
- Only emit relationships where both endpoints appear in your entity list.

Return JSON exactly in this shape:
{{"entities":[{{"name":"...","type":"...","description":"one short sentence"}}],
  "relationships":[{{"source":"...","target":"...","type":"...","evidence":"short quote"}}]}}

TEXT:
\"\"\"{text}\"\"\""""

SCHEMA_CYPHER = [
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT entity_key IF NOT EXISTS FOR (e:Entity) REQUIRE e.key IS UNIQUE",
]


def norm(name: str) -> str:
    return " ".join(name.strip().lower().split())


class GraphBuilder:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
        )

    def close(self):
        self.driver.close()

    def reset(self):
        with self.driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n")
            for stmt in SCHEMA_CYPHER:
                s.run(stmt)
            s.run(
                """CREATE VECTOR INDEX chunk_vec IF NOT EXISTS
                   FOR (c:Chunk) ON (c.embedding)
                   OPTIONS {indexConfig: {
                     `vector.dimensions`: $d,
                     `vector.similarity_function`: 'cosine'}}""",
                d=config.EMBED_DIM,
            )
        print("  schema + vector index ready")

    def add_chunk(self, doc_id, text, emb):
        with self.driver.session() as s:
            s.run(
                "MERGE (c:Chunk {id:$id}) SET c.text=$t, c.embedding=$e",
                id=doc_id, t=text, e=emb,
            )

    def add_extraction(self, doc_id, data):
        ents = data.get("entities", [])
        rels = data.get("relationships", [])
        with self.driver.session() as s:
            for e in ents:
                key = norm(e["name"])
                s.run(
                    """MERGE (n:Entity {key:$key})
                       ON CREATE SET n.name=$name, n.type=$type, n.description=$desc
                       ON MATCH  SET n.description =
                            CASE WHEN size(coalesce(n.description,'')) < size($desc)
                                 THEN $desc ELSE n.description END
                       WITH n MATCH (c:Chunk {id:$doc})
                       MERGE (c)-[:MENTIONS]->(n)""",
                    key=key, name=e["name"], type=e.get("type", "Unknown"),
                    desc=e.get("description", ""), doc=doc_id,
                )
            known = {norm(e["name"]) for e in ents}
            for r in rels:
                sk, tk = norm(r.get("source", "")), norm(r.get("target", ""))
                if sk not in known or tk not in known or sk == tk:
                    continue
                s.run(
                    """MATCH (a:Entity {key:$sk}), (b:Entity {key:$tk})
                       MERGE (a)-[r:RELATES_TO {type:$rt}]->(b)
                       SET r.evidence=$ev, r.doc=$doc""",
                    sk=sk, tk=tk, rt=r.get("type", "RELATED"),
                    ev=r.get("evidence", "")[:300], doc=doc_id,
                )
        return len(ents), len(rels)

    def embed_entities(self):
        """Give every Entity a text embedding = its own name/type/description
        plus the chunks that mention it. These become GraphSAGE input features."""
        with self.driver.session() as s:
            rows = s.run(
                """MATCH (e:Entity)
                   OPTIONAL MATCH (c:Chunk)-[:MENTIONS]->(e)
                   RETURN e.key AS key, e.name AS name, e.type AS type,
                          coalesce(e.description,'') AS d,
                          collect(c.text)[..3] AS ctx"""
            ).data()
        for i, r in enumerate(rows, 1):
            text = f"{r['name']} ({r['type']}). {r['d']} " + " ".join(r["ctx"])
            emb = bedrock.embed(text)
            with self.driver.session() as s:
                s.run("MATCH (e:Entity {key:$k}) SET e.embedding=$e",
                      k=r["key"], e=emb)
            print(f"  entity embedding {i}/{len(rows)}: {r['name']}")

    def stats(self):
        with self.driver.session() as s:
            return s.run(
                """MATCH (c:Chunk) WITH count(c) AS chunks
                   MATCH (e:Entity) WITH chunks, count(e) AS entities
                   MATCH ()-[r:RELATES_TO]->() WITH chunks, entities, count(r) AS rels
                   MATCH ()-[m:MENTIONS]->() RETURN chunks, entities, rels,
                          count(m) AS mentions"""
            ).single().data()


def main():
    gb = GraphBuilder()
    print("Resetting graph...")
    gb.reset()

    docs = get_docs()
    for i, (doc_id, text) in enumerate(docs, 1):
        print(f"[{i}/{len(docs)}] {doc_id}")
        gb.add_chunk(doc_id, text, bedrock.embed(text))
        data = bedrock.chat_json(EXTRACTION_PROMPT.format(text=text),
                                 system=EXTRACTION_SYSTEM)
        n_e, n_r = gb.add_extraction(doc_id, data)
        print(f"     entities={n_e} relationships={n_r}")

    print("Embedding entities...")
    gb.embed_entities()
    print("Graph stats:", json.dumps(gb.stats(), indent=2))
    gb.close()


if __name__ == "__main__":
    main()
