import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

st.set_page_config(page_title="PDF Q&A Chatbot", page_icon="🤖", layout="wide")
st.title("PDF Q&A with Ollama + FAISS")
st.markdown("Upload a PDF and ask questions about its content!")

model_choice = st.sidebar.selectbox("Choose LLM", ["llama3.2:3b", "gemma3:1b", "llama3.2:1b-instruct-q8_0"])
pdf_files = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

if pdf_files:
    all_text = ""
    for pdf_file in pdf_files:
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                all_text += text +"\n"
        
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(all_text)
    
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    db = FAISS.from_texts(chunks, embedding_model)
    
    
    llm = OllamaLLM(model=model_choice)
    template = """
    You are a helpful assistant. Use only the provided context to answer.
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:
    """
    prompt = PromptTemplate(template=template, input_variabled=["context", "question"])
    st.subheader("Ask Questions about the document")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    query = st.text_input("Your Question:")
    if st.button("Ask") and query:
        baseline_answer = llm(query)
        
        docs = db.similarity_search(query, k=3)
        context = "\n\n".join([d.page_content for d in docs])
        rag_template = prompt.format(context=context, question=query)
        rag_answer = llm(rag_template)
        
        st.session_state.chat_history.append(("You", query))
        st.session_state.chat_history.append(("Bot(Baseline)", baseline_answer))
        st.session_state.chat_history.append(("Bot(RAG)", rag_answer))
        
    print(st.session_state.chat_history)
    for speaker, message in st.session_state.chat_history:
        if "Baseline" in speaker:
            st.markdown(f"🟠 **{speaker}:** {message}")
        elif "RAG" in speaker:
            st.markdown(f"🟢 **{speaker}:** {message}")
        else:
            st.markdown(f"**{speaker}:** {message}")
    
else:
    st.info("Please upload PDFs to begin.")

        
    
    