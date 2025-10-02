# GenAI_Projects

Personal projects to improve myself in the field of Generative AI (GenAI).  
This repository contains multiple experiments and applications exploring LLMs, RAG (Retrieval-Augmented Generation), semantic search, FastAPI, and Streamlit-based apps.  

---

## 📂 Project Structure

### 1. **fastAPI_demo**
- Basic FastAPI setup to implement REST APIs.  

### 2. **call_llm**
- A simple interface to call different LLMs via Ollama  
- Helps in abstracting API calls and testing various model responses. 

### 3. **multi_model_chatbot**
- Chatbot built using FastAPI.  
- Can serve as a backend chatbot service with prompt handling and response streaming supporting multiple models at the run time. 

### 4. **hugging_face**
- Built a chatbot using Hugging Face models.  
- Explores open-source LLMs and Transformers pipelines.  

### 5. **semantic_search_app**
- Application for semantic search over documents.  
- Uses embeddings and similarity search for retrieving relevant context.  
- using chromadb for vector search.

### 6. **RAG_implement**
- Built a question-answering bot for a single PDF/document using Retrieval-Augmented Generation (RAG).  
- Handles chunking, embeddings, and vector search for document queries.
- Uses langchain utilities for chunking, embedding , vector search and llm.
- Uses FAISS for vector search

### 7. **streamlit_QA_app**
- Streamlit-based Question Answering system with multiple modes:  
  - **Single Doc QA** → Question answering from one uploaded document.  
  - **Multi Doc QA** → Supports querying across multiple documents.  
  - **Multi-Model Multi-Doc QA** → Allows switching between multiple models for multi-document QA. 

### 8. **company policy qa app**
- Streamlit-based Question Answering system
- Use PyPDFLoader to load the document
- Using FAISS to store the embeddings in memory.
- Using CromaDB to permanent store the embeddings.

### 9. **News Article Summarizer**
- Build “News Article Summarizer” using LlamaIndex

### 10. **Semantic search on 100+ Wikipedia docs**
- Push embeddings to Pinecone from LangChain.
- Compare FAISS vs Pinecone.

### 11. **Conversation Question Answer from PDF**
- Load and chunks a PDF
- Embeds chunks into FAISS vector database
- Uses Ollama model for both embeddings + LLM
- Retrieves relevant chunks for each user query
- Keeps chat history for follow up questions
- Runs in a simple command-line chatbot loop.

### 12. **Generate the image**
- Generate the image with the help of diffusion model
- Using Hugging Face Diffuser Libraray
- Using python libraray torch to run the neutal network
---


## Getting Started

### Installation
- Clone the repository and install dependencies:
- commands:
  - git clone https://github.com/Twinkle16Mittal/GenAI_Projects.git
  - cd GenAI_Projects
  - pip install -r requirements.txt
  - to run the streamlit app -> streamlit run <file_name.py>
