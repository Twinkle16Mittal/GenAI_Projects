import ollama
import chromadb


client = chromadb.Client()

# Define your documents
docs = [
    "Twinkle is very great girl.",
    "She has completed her education.",
    "Twinkle is living in Saharanpur."
]

collection = client.create_collection("documents")
for doc in docs:
    embedding_response = ollama.embed(model="nomic-embed-text", input=doc)
    embedding = embedding_response["embeddings"][0]
    collection.add(
        documents=[doc],
        metadatas=[{"source":"Wikipedia"}],
        ids=[str(hash(doc))],
        embeddings=[embedding]
    )
    
query = "Where Twinkle is living?"
collection = client.get_collection("documents")
query_response = ollama.embed(model="nomic-embed-text", input=query)
query_embedding = query_response["embeddings"][0]
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=1
)

for result in results["documents"][0]:
    print(result)