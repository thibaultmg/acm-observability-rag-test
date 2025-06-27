# main.py
import uvicorn
import time
import os
import sys
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Literal, Optional

# --- LlamaIndex Core Imports ---
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

print("--- Starting RAG API Server (Simplified Storage) ---")

# --- 1. Global Settings and Configuration ---
# Configure LlamaIndex to use local models served by Ollama
# We are using gemma2 for generation and nomic-embed-text for embeddings
print("Configuring LlamaIndex settings...")
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
Settings.llm = Ollama(model="granite3.2:8b", request_timeout=180.0)

# --- 2. Load Documents and Build/Load the Index (Simplified) ---
# This section uses LlamaIndex's default file-based storage.

# Define the paths for data and the persistent index
DATA_DIR = "./data"
PERSIST_DIR = "./storage"

# Ensure the data directory exists
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    print(f"Created data directory at {DATA_DIR}. Please add your documents here.")

# Load the index from disk if it exists
if os.path.exists(PERSIST_DIR):
    print(f"Loading existing index from {PERSIST_DIR}...")
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
    print("Index loaded successfully.")
else:
    print("Building new index from documents...")
    # If the index doesn't exist, load documents and build it
    try:
        documents = SimpleDirectoryReader(DATA_DIR).load_data()
        if not documents:
            print("No documents found in the data directory. The RAG system will only use the LLM's base knowledge.")
            index = None
        else:
            # Create the index from documents
            index = VectorStoreIndex.from_documents(documents)
            # Persist the index to disk for future runs
            index.storage_context.persist(persist_dir=PERSIST_DIR)
            print(f"New index built and saved to {PERSIST_DIR}.")
    except Exception as e:
        print(f"Error loading documents or building index: {e}")
        print("Exiting. Please check your data folder and dependencies.")
        sys.exit(1)

# Create the query engine if the index was loaded/created successfully
query_engine = index.as_query_engine(streaming=False) if index else None
if query_engine:
    print("RAG query engine is ready.")
else:
    print("RAG query engine could not be created. API will not be functional.")
    sys.exit(1)


# --- 3. Define OpenAI-compatible Pydantic Models for the API ---
# These models enforce the structure of the API requests and responses,
# ensuring compatibility with any client designed for OpenAI.
class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

class ChatCompletionRequest(BaseModel):
    model: str # Not used by our backend but required by the OpenAI spec
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.1
    stream: Optional[bool] = False # We will not support streaming in this basic example

class ChatCompletionResponseChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Literal["stop", "length"] = "stop"

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: "chatcmpl-" + str(time.time()))
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "ollama-gemma2-rag"
    choices: List[ChatCompletionResponseChoice]


# --- 4. Create the FastAPI Application and Endpoints ---
app = FastAPI()

@app.get("/health", summary="Health Check")
def health_check():
    """Simple health check endpoint to confirm the server is running."""
    return {"status": "ok"}

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse, summary="OpenAI-compatible Chat Endpoint")
def chat_completions(request: ChatCompletionRequest):
    """
    This endpoint mimics the OpenAI /v1/chat/completions API.
    It takes a list of messages and returns a response from the RAG system.
    """
    if not query_engine:
        raise HTTPException(status_code=503, detail="RAG query engine is not available.")
    
    # We only consider the last user message for this simple RAG implementation
    user_message = next((msg.content for msg in reversed(request.messages) if msg.role == 'user'), None)
    
    if not user_message:
        raise HTTPException(status_code=400, detail="No user message found in the request.")

    # Log the incoming query with a timestamp
    print(f"[{datetime.now().isoformat()}] Received query: '{user_message}'")
    
    # Query the RAG engine to get a response
    rag_response = query_engine.query(user_message)
    
    # Format the response into the OpenAI-compatible structure
    assistant_message = ChatMessage(role="assistant", content=str(rag_response))
    choice = ChatCompletionResponseChoice(message=assistant_message)
    response = ChatCompletionResponse(choices=[choice])
    
    # Log the outgoing response with a timestamp
    # .model_dump_json() is used for a clean, indented JSON representation of the response
    print(f"[{datetime.now().isoformat()}] Sending response: {response.model_dump_json(indent=2)}")

    return response

# --- 5. Main Execution Block ---
if __name__ == "__main__":
    # To run this script, save it as `main.py` and execute `uvicorn main:app --reload`
    print("Starting Uvicorn server at http://localhost:8000")
    print("Access the API documentation at http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
