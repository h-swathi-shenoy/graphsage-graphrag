"""Central config: Neo4j connection + AWS Bedrock model IDs."""
import os

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "graphsage123")

AWS_REGION = os.getenv("AWS_REGION", "<region-name>")
EMBED_MODEL = os.getenv("BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v2:0")
CHAT_MODEL = os.getenv("BEDROCK_CHAT_MODEL", "us.amazon.nova-lite-v1:0")
EMBED_DIM = 1024

# GraphSAGE hyperparameters
SAGE_HIDDEN = 256
SAGE_OUT = 128
SAGE_LAYERS = 2
SAGE_EPOCHS = 400
SAGE_LR = 0.01
SAGE_FANOUT = 10          # neighbours sampled per layer
WALK_LENGTH = 4           # random-walk length for positive pairs
WALKS_PER_NODE = 10
NEG_SAMPLES = 5
