"""Thin AWS Bedrock wrapper: Titan embeddings + Nova chat (JSON-safe)."""
import json
import re
import time
from typing import List

import boto3
from botocore.config import Config

import config

_cfg = Config(retries={"max_attempts": 8, "mode": "adaptive"}, read_timeout=120)
_rt = boto3.client("bedrock-runtime", region_name=config.AWS_REGION, config=_cfg)


def embed(text: str) -> List[float]:
    """Embed one string with Titan Text Embeddings V2 -> 1024-d vector."""
    body = json.dumps({"inputText": text[:8000], "dimensions": config.EMBED_DIM,
                       "normalize": True})
    for attempt in range(5):
        try:
            r = _rt.invoke_model(modelId=config.EMBED_MODEL, body=body)
            return json.loads(r["body"].read())["embedding"]
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def embed_many(texts: List[str]) -> List[List[float]]:
    return [embed(t) for t in texts]


def chat(prompt: str, system: str = "", max_tokens: int = 1200,
         temperature: float = 0.0) -> str:
    """Single-turn call to a Bedrock Converse-compatible model."""
    kwargs = dict(
        modelId=config.CHAT_MODEL,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
    )
    if system:
        kwargs["system"] = [{"text": system}]
    for attempt in range(5):
        try:
            r = _rt.converse(**kwargs)
            return r["output"]["message"]["content"][0]["text"]
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def chat_json(prompt: str, system: str = "") -> dict:
    """Chat call that must return a JSON object; tolerates markdown fences."""
    raw = chat(prompt, system=system)
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        raise ValueError(f"No JSON found in model output:\n{raw[:400]}")
    return json.loads(m.group(0))
