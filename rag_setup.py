"""
One-time setup: embeds your portfolio case studies and upserts them into Pinecone.
Re-run this whenever you add or update a case study in case_studies.py.

pip install pinecone sentence-transformers
export PINECONE_API_KEY=your_pinecone_key
python rag_setup.py

Uses Pinecone's free Starter plan (5 serverless indexes, 2GB storage - one small
index with 7 case studies uses a tiny fraction of that) and a free LOCAL embedding
model (no OpenAI/embedding API cost).
"""

import os
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from case_studies import CASE_STUDIES

INDEX_NAME = "sales-copilot-case-studies"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # free, local, 384-dim, ~80MB download
EMBED_DIM = 384

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

existing = [idx["name"] for idx in pc.list_indexes()]
if INDEX_NAME not in existing:
    print(f"Creating Pinecone index '{INDEX_NAME}'...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBED_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
else:
    print(f"Index '{INDEX_NAME}' already exists, reusing it.")

index = pc.Index(INDEX_NAME)
model = SentenceTransformer(EMBED_MODEL_NAME)

vectors = []
for cs in CASE_STUDIES:
    text = f"{cs['title']} - {cs['industry']} - {cs['summary']}"
    embedding = model.encode(text).tolist()
    vectors.append(
        {
            "id": cs["id"],
            "values": embedding,
            "metadata": {
                "title": cs["title"],
                "industry": cs["industry"],
                "summary": cs["summary"],
            },
        }
    )

index.upsert(vectors=vectors)
print(f"Upserted {len(vectors)} case studies into Pinecone index '{INDEX_NAME}'.")
