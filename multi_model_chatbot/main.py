from fastapi import FastAPI
from pydantic import BaseModel
import ollama

app = FastAPI(title="Multi-Model Chatbot")

# Available models
MODELS = {
    "llama3.2:3b": "llama3.2:3b",
    "gemma3:1b" : "gemma3:1b",
    "llama3.2:1b-instruct-q8_0" : "llama3.2:1b-instruct-q8_0"
}

conversations = {model:[] for model in MODELS}

class ChatResponse(BaseModel):
    model: str
    message: str
    
@app.get("/all_models")
async def root():
    return {"available_models": list(MODELS.keys())}

@app.post("/chat")
async def chat(req: ChatResponse):
    model_name = req.model
    if model_name not in MODELS:
        return {"error":f"Invalid model. Choose from {list(MODELS.keys())}"}
    
    conversations[model_name].append({"role":"user", "content":req.message})
    response = ollama.chat(model=model_name, messages=conversations[model_name])
    reply = response["message"]["content"]
    
    #Add Assistant reply
    conversations[model_name].append({"role":"assistant", "content":reply})
    return {"model_name": model_name, "reply": reply}