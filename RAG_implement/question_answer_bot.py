from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
# vector serach Library FAISS
from langchain_community.vectorstores import FAISS

from langchain_ollama import OllamaLLM
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


#Load Pdf
reader = PdfReader("sample.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text() + "\n"
    
# Split into Chunks

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(text)
print(f"Total chunks:{len(chunks)}")

# Embedding model (Ollama)
embedding_model = OllamaEmbeddings(model="nomic-embed-text")

# store in FAISS
db = FAISS.from_texts(chunks, embedding_model)

# llm model
llm = OllamaLLM(model="llama3.2:3b")
# query = "Summarize the key findings of the PDF."
# docs = db.similarity_search(query, k=3)

# #Build Context
# context = "\n\n".join([d.page_content for d in docs])

# # print(context)

# prompt = f"""
# Answer the question based ONLY on the text below. 
# Do not make assumptions, and do not mention if a PDF is missing.

# Document content:
# {context}

# Question: {query}
# """
# response = llm(prompt)
# print("Answer", response)

while True:
    query = input("Ask a question (or 'exit'): ")
    if query.lower() == "exit":
        break
    docs = db.similarity_search(query, k=3)
    context = "\n\n".join([d.page_content for d in docs])
    prompt = f"Answer based on the document:\n\n{context}\n\nQuestion: {query}"
    print("Answer:", llm(prompt))

