import os, time
import wikipedia
import numpy as np
import faiss
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.embeddings import OllamaEmbeddings
from pinecone import Pinecone, ServerlessSpec

# Load .env
load_dotenv()
print("PINECONE_API_KEY:", os.getenv("PINECONE_API_KEY"))

NUM_PAGES = 120
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
EMBEDDING_MODEL = "nomic-embed-text"
PINECONE_INDEX_NAME = "wiki-demo-ollama"

# fetch Wikipedia
print(f"fetching {NUM_PAGES} random Wikipedia pages...")
titles = wikipedia.random(pages=NUM_PAGES)
raw_docs = {}
for t in titles:
    try:
        page = wikipedia.page(t, auto_suggest=False)
        raw_docs[t] = page.content
    except Exception:
        try:
            raw_docs[t] = wikipedia.summary(t)
        except:
            pass
print(f"Fetched {len(raw_docs)} pages.")

# Chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
documents = []
for title, text in raw_docs.items():
    chunks = splitter.split_text(text)
    for i, c in enumerate(chunks):
        documents.append({
            "title": title,
            "page_content": c,
            "metadata": {"title": title, "chunk": i}
        })
print(f"Total chunks: {len(documents)}")

# Ollama Embeddings
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
texts = [d["page_content"] for d in documents]

print("Embedding chunks with Ollama...")
t0 = time.perf_counter()
vectors = embeddings.embed_documents(texts)
t1 = time.perf_counter()
print(f"Embeded {len(vectors)} vectors in {t1-t0:.2f}s")

# ---------------------------
# ✅ Pinecone new client usage
# ---------------------------
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Create index if not exists
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=len(vectors[0]),
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(PINECONE_INDEX_NAME)

print("Upserting to Pinecone...")
BATCH_SIZE = 16
for i in range(0, len(vectors), BATCH_SIZE):
    batch_ids = [f"doc-{j}" for j in range(i, min(i+BATCH_SIZE, len(vectors)))]
    batch_vecs = vectors[i:i+BATCH_SIZE]
    batch_meta = [documents[j]["metadata"] for j in range(i, min(i+BATCH_SIZE, len(vectors)))]
    upserts = [(batch_ids[k], batch_vecs[k], batch_meta[k]) for k in range(len(batch_ids))]
    index.upsert(vectors=upserts)

print("Upsert Complete.")

# FAISS
print("Building FAISS index locally...")
vecs_np = np.array(vectors, dtype=np.float32)

# normalize for cosine similarity
norms = np.linalg.norm(vecs_np, axis=1, keepdims=True)
vecs_np = vecs_np / norms
faiss_index = faiss.IndexFlatIP(vecs_np.shape[1])
faiss_index.add(vecs_np)

# Query Both
query = "What are the causes and treatments of diabetes?"
print(f"\nQuery: {query}")

q_vec = embeddings.embed_query(query)
q_vec_np = np.array([q_vec], dtype=np.float32)
q_vec_np = q_vec_np / np.linalg.norm(q_vec_np, axis=1, keepdims=True)

# Pinecone Query
t0 = time.perf_counter()
pc_res = index.query(vector=q_vec, top_k=5, include_metadata=True)
t1 = time.perf_counter()
print(f"\nPinecone results ({t1-t0:.4f}s):")
for m in pc_res["matches"]:
    print(f"- {m['metadata']['title']} (score {m['score']:.4f})")

# FAISS Query
t0 = time.perf_counter()
D, I = faiss_index.search(q_vec_np, 5)
t1 = time.perf_counter()
print(f"\nFAISS results ({t1-t0:.4f}s):")
for idx, score in zip(I[0], D[0]):
    print(f"- {documents[idx]['title']} (score {score:.4f})")
