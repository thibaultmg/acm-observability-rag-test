# main.py
import uvicorn
import time
import os
import sys
import argparse
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
# --- UPDATED: Import LlamaIndex's ChatMessage for type compatibility ---
from llama_index.core.llms import ChatMessage as LlamaChatMessage
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

# --- UPDATED: Use a ChatEngine instead of a QueryEngine ---
# A ChatEngine is designed for conversational, multi-turn interactions.
chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=False) if index else None
if chat_engine:
    print("RAG chat engine is ready.")
else:
    print("RAG chat engine could not be created.", file=sys.stderr)


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
    if not chat_engine:
        raise HTTPException(status_code=503, detail="RAG chat engine is not available.")
    
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages found in the request.")

    # --- UPDATED: Handle conversation history ---
    # Convert Pydantic models to LlamaIndex's ChatMessage models
    all_messages = [LlamaChatMessage(role=msg.role, content=msg.content) for msg in request.messages]
    
    # Separate the last message from the history
    last_message = all_messages[-1]
    chat_history = all_messages[:-1]
    
    print(f"[{datetime.now().isoformat()}] Received chat. History length: {len(chat_history)}, Query: '{last_message.content}'")

    # The chat engine's `chat` method can take a history.
    # For a stateless API, we pass the history with each request.
    # NOTE: The default ChatEngine is not thread-safe. For production, you'd need
    # to manage engine instances per user/session.
    response = chat_engine.chat(last_message.content, chat_history=chat_history)
    
    assistant_message = ChatMessage(role="assistant", content=str(response.response))
    choice = ChatCompletionResponseChoice(message=assistant_message)
    response_payload = ChatCompletionResponse(choices=[choice])
    
    print(f"[{datetime.now().isoformat()}] Sending response: {response_payload.model_dump_json(indent=2)}")
    return response_payload

# --- 5. Main Execution Block (with mode selection) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the RAG system as a server or in interactive chat mode.")
    parser.add_argument("--chat", action="store_true", help="Run in interactive chat mode in the terminal.")
    args = parser.parse_args()

    if args.chat:
        # --- Run Interactive Chat Mode ---
        if not chat_engine:
            print("Chat engine not available. Exiting chat mode.", file=sys.stderr)
            sys.exit(1)
            
        print("\n--- Starting Interactive Chat Mode ---")
        print("Type 'exit' or 'quit' to end the session. Type 'reset' to clear conversation history.")
        
        # In chat mode, the engine's internal memory provides statefulness.
        # We reset it at the start of a new session.
        chat_engine.reset()
        
        while True:
            try:
                user_message = input("\nYou: ")
                if user_message.lower() in ["exit", "quit"]:
                    print("Exiting chat mode. Goodbye!")
                    break
                
                if user_message.lower() == "reset":
                    chat_engine.reset()
                    print("...Conversation history has been reset...")
                    continue

                if not user_message.strip():
                    continue

                print("Bot: Thinking...")
                # The chat engine automatically uses its internal memory
                response = chat_engine.chat(user_message)
                # The response object has a `.response` attribute with the text
                print(f"\nBot: {response.response}")

            except KeyboardInterrupt:
                print("\nExiting chat mode. Goodbye!")
                break
            except Exception as e:
                print(f"An error occurred: {e}", file=sys.stderr)
    else:
        # --- Run Server Mode (Default) ---
        if not chat_engine:
            print("Chat engine not available. Cannot start server.", file=sys.stderr)
            sys.exit(1)
        print("Starting Uvicorn server at http://localhost:8000")
        print("Access the API documentation at http://localhost:8000/docs")
        print("To inspect chunks, visit http://localhost:8000/chunks")
        uvicorn.run(app, host="0.0.0.0", port=8000)
