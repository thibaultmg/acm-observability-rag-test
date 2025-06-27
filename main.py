# main.py
import uvicorn
import time
import os
import sys
import argparse # <-- Added for command-line flag parsing
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
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import TextNode
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

print("--- Starting RAG API Server with Hybrid Chunking Strategy ---")

# --- 1. Global Settings and Configuration ---
print("Configuring LlamaIndex settings...")
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
Settings.llm = Ollama(model="granite3.3:latest", request_timeout=180.0)

# Define the semantic splitter for general use
semantic_node_parser = SemanticSplitterNodeParser(
    embed_model=Settings.embed_model, 
    breakpoint_percentile_threshold=95
)
print(f"SemanticSplitter is configured and will be used for non-FAQ files.")


# --- 2. Load Documents and Build/Load the Index ---
DATA_DIR = "./data"
PERSIST_DIR = "./storage"

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    print(f"Created data directory at {DATA_DIR}. Please add your documents here.")

# This logic must run for both server and chat mode
# so we keep it in the main script body.
if os.path.exists(PERSIST_DIR):
    print(f"Loading existing index from {PERSIST_DIR}...")
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
    print("Index loaded successfully.")
else:
    print("Building new index with hybrid chunking strategy...")
    try:
        nodes = []
        for filename in os.listdir(DATA_DIR):
            file_path = os.path.join(DATA_DIR, filename)
            if not os.path.isfile(file_path):
                continue

            if filename.endswith("_faq.md"):
                print(f"Applying custom '---' splitter for: {filename}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                chunks = content.split("\n---\n")
                for i, chunk_text in enumerate(chunks):
                    if chunk_text.strip():
                        node = TextNode(
                            text=chunk_text.strip(),
                            metadata={"file_name": filename, "chunk_number": i}
                        )
                        nodes.append(node)
            else:
                print(f"Applying semantic splitter for: {filename}")
                documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
                split_nodes = semantic_node_parser.get_nodes_from_documents(documents)
                nodes.extend(split_nodes)

        if not nodes:
            print("No documents found. The RAG system will only use the LLM's base knowledge.")
            index = None
        else:
            index = VectorStoreIndex(nodes)
            index.storage_context.persist(persist_dir=PERSIST_DIR)
            print(f"New index built and saved to {PERSIST_DIR}.")

    except Exception as e:
        print(f"Error loading documents or building index: {e}", file=sys.stderr)
        sys.exit(1)

query_engine = index.as_query_engine(streaming=False) if index else None
if query_engine:
    print("RAG query engine is ready.")
else:
    print("RAG query engine could not be created.", file=sys.stderr)


# --- 3. Define OpenAI-compatible Pydantic Models (for server mode) ---
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


# --- 4. Create the FastAPI Application (for server mode) ---
app = FastAPI()

@app.get("/health", summary="Health Check")
def health_check():
    return {"status": "ok"}

@app.get("/chunks", summary="Inspect Document Chunks")
def get_chunks():
    if not index:
         raise HTTPException(status_code=503, detail="Index is not available.")
    nodes = index.docstore.docs.values()
    if not nodes:
        return {"message": "No nodes found in the index."}
    chunks_data = [
        {"node_id": node.node_id, "text": node.get_content(), "metadata": node.metadata, "char_count": len(node.get_content())} for node in nodes
    ]
    return Response(content=json.dumps(chunks_data, indent=2), media_type="application/json")

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse, summary="OpenAI-compatible Chat Endpoint")
def chat_completions(request: ChatCompletionRequest):
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

# --- 5. Main Execution Block (with mode selection) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the RAG system as a server or in interactive chat mode.")
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Run in interactive chat mode in the terminal."
    )
    args = parser.parse_args()

    if args.chat:
        # --- Run Interactive Chat Mode ---
        if not query_engine:
            print("Query engine not available. Exiting chat mode.", file=sys.stderr)
            sys.exit(1)
            
        print("\n--- Starting Interactive Chat Mode ---")
        print("Type 'exit' or 'quit' to end the session.")
        
        while True:
            try:
                user_message = input("\nYou: ")
                if user_message.lower() in ["exit", "quit"]:
                    print("Exiting chat mode. Goodbye!")
                    break
                
                if not user_message.strip():
                    continue

                print("Bot: Thinking...")
                response = query_engine.query(user_message)
                print(f"\nBot: {response}")

            except KeyboardInterrupt:
                print("\nExiting chat mode. Goodbye!")
                break
            except Exception as e:
                print(f"An error occurred: {e}", file=sys.stderr)
    else:
        # --- Run Server Mode (Default) ---
        if not query_engine:
            print("Query engine not available. Cannot start server.", file=sys.stderr)
            sys.exit(1)
        print("Starting Uvicorn server at http://localhost:8000")
        print("Access the API documentation at http://localhost:8000/docs")
        print("To inspect chunks, visit http://localhost:8000/chunks")
        uvicorn.run(app, host="0.0.0.0", port=8000)
