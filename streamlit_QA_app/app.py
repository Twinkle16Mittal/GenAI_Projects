import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

st.set_page_config(page_title="PDF Q&A Chatbot", page_icon="🤖", layout="wide")

st.title("PDF Q&A with Ollama + FAISS")
st.markdown("Upload a PDF and ask questions about its content!")

# Sidebar: Upload PDF
pdf_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

if pdf_file:
    # Read PDF
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()+"\n"
        
    # Split into Chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    
    #Embeddings
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    db = FAISS.from_texts(chunks, embedding_model)
    
    #LLM
    llm = OllamaLLM(model="llama3.2:3b")
    
    # Chat UI
    st.subheader("Ask Questions about the document")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    query = st.text_input("Your Question:")
    
    if st.button("Ask") and query:
        # Retrieve docs
        docs = db.similarity_search(query, k=3)
        context = "\n\n".join([d.page_content for d in docs])
        
        #prompt
        prompt = f"""
        Answer the question based only on the uploaded pdf.
        Do not make assumptions.
        
        Document Conext
        {context}
        Question: {query}
        """
        
        answer = llm(prompt)
        st.session_state.chat_history.append(("You", query))
        st.session_state.chat_history.append(("Bot", answer))
        
    # Display chat history
    print(st.session_state.chat_history)
    for speaker, message in st.session_state.chat_history:
        st.markdown(f"{speaker}: {message}")
            
else:
    st.info("Please upload a PDF to begin.")
        