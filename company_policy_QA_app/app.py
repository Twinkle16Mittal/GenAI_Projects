import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS, Chroma
from langchain.prompts import PromptTemplate
import os
import tempfile

# Fix for Mac / Linux duplicate library issue
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Streamlit setup
st.set_page_config(page_title="Company Policy Q&A Chatbot", page_icon="🤖", layout="wide")
st.title("📄 Company Policy Q&A with Ollama + FAISS")
st.markdown("Upload PDFs (HR, Leave, Security) and ask questions about them!")

# Upload PDFs
pdf_files = st.sidebar.file_uploader(
    "Upload Policy PDFs", type=["pdf"], accept_multiple_files=True
)

if pdf_files:
    # Load documents using PyPDFLoader
    docs = []
    for pdf_file in pdf_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.read())
            tmp_file_path = tmp_file.name
        loader = PyPDFLoader(tmp_file_path)
        docs.extend(loader.load())

    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # Embeddings + FAISS
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    # use FAISS
    # db = FAISS.from_documents(chunks, embedding_model)
    
    #use chromaDB
    # only for first time
    # db = Chroma.from_documents(
    # documents=chunks,
    # embedding=embedding_model,
    # persist_directory="chroma_store"    
    # )
    # db.persist()
    # next time
    db = Chroma(
    embedding_function=embedding_model,
    persist_directory="chroma_store"
    )

    # Ollama LLM (change to any local model you have, e.g. mistral, llama2, etc.)
    llm = OllamaLLM(model="llama3.2:3b")

    # Prompt template for RAG
    template = """
    You are a helpful assistant. Use only the provided context to answer.
    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    st.subheader("Ask Questions about the Policies")

    # Maintain chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    query = st.text_input("Your Question:")
    if st.button("Ask") and query:
        # Baseline answer (no retrieval, raw LLM)
        baseline_answer = llm(query)

        # Retrieval step
        docs = db.similarity_search(query, k=3)
        context = "\n\n".join([d.page_content for d in docs])

        # RAG answer (context + query)
        rag_template = prompt.format(context=context, question=query)
        rag_answer = llm(rag_template)

        # Update chat history
        st.session_state.chat_history.append(("You", query))
        st.session_state.chat_history.append(("Bot (Baseline)", baseline_answer))
        st.session_state.chat_history.append(("Bot (RAG)", rag_answer))

    # Display chat history
    for speaker, message in st.session_state.chat_history:
        if "Baseline" in speaker:
            st.markdown(f"🟠 **{speaker}:** {message}")
        elif "RAG" in speaker:
            st.markdown(f"🟢 **{speaker}:** {message}")
        else:
            st.markdown(f"**{speaker}:** {message}")

else:
    st.info("Please upload your HR, Leave, and Security policy PDFs to begin.")
