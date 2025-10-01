import argparse # Command line argument parsing
import os
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
# combines docs + LLM response generation
from langchain.chains.combine_documents import create_stuff_documents_chain
# allows retriever to rephrase follow-up questions using chat history
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# Stores chat history in memory
from langchain.memory import ConversationBufferMemory
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# conversational question-answering chain from a pdf 
def build_qa_chain(pdf_path: str, ollama_model: str="llama3.2:3b",
                   persist_dir: Optional[str] = None,embedding_model_name: str = "nomic-embed-text",
                   chunk_size: int = 1000,chunk_overlap: int = 200,k: int = 4):
    
    # 1 load the document
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"[info] Loaded {len(docs)} page documents from PDF")
    
    # 2 Split into multiple chunks for better retrieval
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                              separators=["\n\n", "\n", " ", ""])
    docs = splitter.split_documents(docs)
    print(f"[info] Split into {len(docs)} chunks (chunk_size={chunk_size}, overlap={chunk_overlap})")
    
    # 3) Embeddings (convert into in numerical values)
    print(f"[info] Loading embeddings model: {embedding_model_name}. This may take a moment.")
    embeddings = OllamaEmbeddings(model=embedding_model_name)
    
    # 4) Create or reuse FAISS vectorstore
    if persist_dir and os.path.exists(persist_dir):
        print(f"[info] Loading FAISS index from {persist_dir}")
        vectordb = FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)
    else:
        print("[info] Creating FAISS index (this may take a moment)...")
        vectordb = FAISS.from_documents(docs, embeddings)
        if persist_dir:
            os.makedirs(persist_dir, exist_ok=True)
            vectordb.save_local(persist_dir)
            print(f"[info] Saved FAISS index to {persist_dir}")
    
    # retrieve from faiss          
    retriever =vectordb.as_retriever(search_type="similarity", search_kwargs={"k":k})
    print(f"[info] Retriever configured (k={k})")

    # load the llm model
    llm = OllamaLLM(model=ollama_model)

    # 6) Prompt Templates
    # rewrites follow up questions into standalone questions using chat history
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Rephrase the user question into a standalone query if it depends on chat history."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # wrpas retriever so that it can handle multi-turn conversation
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    # main prompt for llm to answer the user's question
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant answering questions about a PDF document. "
                   "Use only the provided context. If unsure, say 'I don't know based on the document.'"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        ("system", "Context:\n{context}\nAnswer:")
    ])
    
    # create a chain that stuffs retrieved docs + prompt into LLM.
    doc_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # connects retriever + QA chain into a pipeline
    retrieval_chain = create_retrieval_chain(history_aware_retriever, doc_chain)
    # Memory
    memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

    return retrieval_chain, memory, vectordb

def run_cli_loop(chain, memory):
    print("\n=== Conversational PDF Assistant (type 'exit' or 'quit' to stop) ===\n")
    while True:
        user_q = input("You: ").strip()
        if user_q.lower() in ("exit","quit"):
            print("Exiting. Bye!")
            break

        # run the chain
        result = chain.invoke({"input": user_q, "chat_history": memory.load_memory_variables({})["chat_history"]})
        answer = result["answer"]
        
        print("\n Assistant:", answer.strip(), "\n")
        print("--\n")

        # save conversation into memory
        memory.save_context({"input": user_q}, {"output": answer})
        
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Conversational PDF Assistant using LangChain + Ollama")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file")
    parser.add_argument("--model", type=str, default="llama3.2:3b", help="Ollama model name (as served locally)")
    parser.add_argument("--persist_dir", type=str, default=None,help="Optional directory to persist FAISS index")
    parser.add_argument("--k", type=int, default=4, help="Retriever top-k")
    args = parser.parse_args()
    
    chain, memory, vectordb = build_qa_chain(
        pdf_path=args.pdf_path,
        ollama_model=args.model,
        persist_dir=args.persist_dir,
        k=args.k
    )
    
    run_cli_loop(chain, memory)
                
        

    
     
    