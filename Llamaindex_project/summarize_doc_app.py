from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

#: Load you LLM(Ollama)
llm = Ollama(model="gemma3:1b")

# Load local embedding instead of openai because llamaindex bydefault uses openai embeddings
ollama_embedding = OllamaEmbedding(
    model_name="nomic-embed-text")
        
documents = SimpleDirectoryReader("news_data").load_data()

# Build an index
index = VectorStoreIndex.from_documents(documents,embed_model=ollama_embedding)

# Create a Query Engine
query_engine = index.as_query_engine(llm=llm)

# Ask for a summary
response = query_engine.query("Summarize this news article in 5 bullet points.")
print(response)