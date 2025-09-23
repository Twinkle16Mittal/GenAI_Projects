from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import ollama
import requests

app = FastAPI()


class ArticleRequest(BaseModel):
    text: str
    
def check_ollama_running():
    """
    Check if Ollama local server is reachable.
    """
    try:
        r = requests.get("http://localhost:11434/v1/models", timeout=2)
        return r.status_code == 200
    except requests.RequestException:
        return False
    
@app.post("/summarize")
async def summarize_article(article: ArticleRequest):
    # call GPT-4o-mini
    if not check_ollama_running():
        raise HTTPException(status_code=503,
                            detail="Ollama server is not running. Please start Ollama using ollama serve.")
    response = ollama.chat(
        model="llama3.2:1b-instruct-q8_0",
         messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes articles."},
            {"role": "user", "content": f"Summarize the following article:\n\n{article.text}"}
        ]
    )
    summary = response["message"]["content"]
    return {"summary": summary} 