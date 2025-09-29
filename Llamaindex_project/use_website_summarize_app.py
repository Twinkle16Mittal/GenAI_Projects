import requests
from bs4 import BeautifulSoup
from llama_index.core import Document
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
#: Load you LLM(Ollama)
llm = Ollama(model="gemma3:1b")

# Load local embedding instead of openai because llamaindex bydefault uses openai embeddings
ollama_embedding = OllamaEmbedding(
    model_name="nomic-embed-text")
# Fetch and parse
url = "https://www.bbc.com/news/world-123456"  # example
html = requests.get(url).text
soup = BeautifulSoup(html, "html.parser")
article_text = " ".join([p.get_text() for p in soup.find_all("p")])

# Create a Document
doc = Document(text=article_text)

# Index and summarize
index = VectorStoreIndex.from_documents([doc], embed_model=ollama_embedding)
query_engine = index.as_query_engine(llm=llm)
response = query_engine.query("Summarize this news article in 5 bullet points.")
print(response)
