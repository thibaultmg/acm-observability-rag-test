# main.py
import uvicorn
import time
import os
import sys
from datetime import datetime
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
import json

# --- LlamaIndex Core Imports ---
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
)
# --- UPDATED: Import SemanticSplitterNodeParser ---
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

print("--- Starting RAG API Server with Chunk Inspection ---")

# --- 1. Global Settings and Configuration ---
print("Configuring LlamaIndex settings...")
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
Settings.llm = Ollama(model="granite3.2:8b", request_timeout=180.0)

# --- UPDATED: Using a semantic chunker ---
# SemanticSplitterNodeParser attempts to split text based on semantic meaning,
# which can be more effective than fixed-size chunks. It uses the embedding model
# to find semantic breakpoints.
# breakpoint_percentile_threshold: Lower values create more, smaller chunks. Higher values create fewer, larger chunks.
node_parser = SemanticSplitterNodeParser(
    embed_model=Settings.embed_model, 
    breakpoint_percentile_threshold=95
)
Settings.transformations = [node_parser]
print(f"Using SemanticSplitterNodeParser with breakpoint_percentile_threshold={node_parser.breakpoint_percentile_threshold}")


# --- 2. Load Documents and Build/Load the Index ---
DATA_DIR = "./data"
PERSIST_DIR = "./storage"

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    print(f"Created data directory at {DATA_DIR}. Please add your documents here.")

if os.path.exists(PERSIST_DIR):
    print(f"Loading existing index from {PERSIST_DIR}...")
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
    print("Index loaded successfully.")
else:
    print("Building new index from documents...")
    try:
        documents = SimpleDirectoryReader(DATA_DIR).load_data()
        if not documents:
            print("No documents found in the data directory. The RAG system will only use the LLM's base knowledge.")
            index = None
        else:
            index = VectorStoreIndex.from_documents(documents)
            index.storage_context.persist(persist_dir=PERSIST_DIR)
            print(f"New index built and saved to {PERSIST_DIR}.")
    except Exception as e:
        print(f"Error loading documents or building index: {e}")
        print("Exiting. Please check your data folder and dependencies.")
        sys.exit(1)

query_engine = index.as_query_engine(streaming=False) if index else None
if query_engine:
    print("RAG query engine is ready.")
else:
    print("RAG query engine could not be created. API will not be functional.")
    sys.exit(1)


# --- 3. Define OpenAI-compatible Pydantic Models ---
class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.1
    stream: Optional[bool] = False

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

# --- UPDATED: Using the more direct .docs.values() method ---
@app.get("/chunks", summary="Inspect Document Chunks")
def get_chunks():
    """
    Retrieves all the text chunks (nodes) from the vector store.
    This is useful for debugging and optimizing the chunking strategy.
    """
    if not index:
         raise HTTPException(status_code=503, detail="Index is not available.")
    
    # Use the more direct .docs.values() method to get all nodes.
    nodes = index.docstore.docs.values()
    
    if not nodes:
        return {"message": "No nodes found in the index."}

    chunks_data = [
        {
            "node_id": node.node_id,
            "text": node.get_content(),
            "metadata": node.metadata,
            "char_count": len(node.get_content())
        }
        for node in nodes
    ]

    return Response(content=json.dumps(chunks_data, indent=2), media_type="application/json")


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse, summary="OpenAI-compatible Chat Endpoint")
def chat_completions(request: ChatCompletionRequest):
    """
    This endpoint mimics the OpenAI /v1/chat/completions API.
    """
    if not query_engine:
        raise HTTPException(status_code=503, detail="RAG query engine is not available.")
    
    user_message = next((msg.content for msg in reversed(request.messages) if msg.role == 'user'), None)
    
    if not user_message:
        raise HTTPException(status_code=400, detail="No user message found in the request.")

    print(f"[{datetime.now().isoformat()}] Received query: '{user_message}'")
    
    rag_response = query_engine.query(user_message)
    
    assistant_message = ChatMessage(role="assistant", content=str(rag_response))
    choice = ChatCompletionResponseChoice(message=assistant_message)
    response = ChatCompletionResponse(choices=[choice])
    
    print(f"[{datetime.now().isoformat()}] Sending response: {response.model_dump_json(indent=2)}")

    return response

# --- 5. Main Execution Block ---
if __name__ == "__main__":
    print("Starting Uvicorn server at http://localhost:8000")
    print("Access the API documentation at http://localhost:8000/docs")
    print("To inspect chunks, visit http://localhost:8000/chunks")
    uvicorn.run(app, host="0.0.0.0", port=8000)
