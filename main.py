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
from llama_index.core.llms import ChatMessage as LlamaChatMessage, LLM

# --- LlamaIndex LLM and Embedding Imports ---
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.google_genai import GoogleGenAI


def initialize_llm(provider, model_name, api_key=None):
    """Initializes and returns the appropriate LLM based on the provider."""
    if provider == "ollama":
        print(f"Initializing Ollama with model: {model_name}")
        return Ollama(model=model_name, request_timeout=180.0)
    elif provider == "gemini":
        print(f"Initializing Gemini with model: {model_name}")
        if not os.getenv("GOOGLE_API_KEY") and not api_key:
             print("Warning: GOOGLE_API_KEY environment variable not set. Gemini may fail to initialize.", file=sys.stderr)
        return GoogleGenAI(model=model_name, api_key=api_key, max_tokens=2048)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def log_retrieved_chunks(source_nodes, query, rewritten_query=None):
    """Logs the retrieved source nodes for a given query to a file."""
    if not source_nodes:
        return
    
    with open("retrieval.log", "a", encoding="utf-8") as f:
        log_entry = f"--- Original Query: {query.strip()} ---\n"
        if rewritten_query:
            log_entry += f"--- Rewritten Query: {rewritten_query.strip()} ---\n"
        log_entry += f"Timestamp: {datetime.now().isoformat()}\n"
        log_entry += f"Retrieved {len(source_nodes)} chunks:\n\n"
        
        for i, node_with_score in enumerate(source_nodes):
            node = node_with_score.node
            score = node_with_score.score
            log_entry += f"Chunk {i+1} (Score: {score:.4f}):\n"
            log_entry += f"  Source: {node.metadata.get('file_name', 'N/A')}\n"
            content = node.get_content().strip().replace('\n', ' ')
            log_entry += f"  Content: {content}\n\n"
        
        log_entry += "=" * 40 + "\n\n"
        f.write(log_entry)
    print(f"Logged {len(source_nodes)} retrieved chunks to retrieval.log")


# --- NEW: Query Rewriting Logic ---
QUERY_REWRITE_PROMPT_TMPL = """Given the following conversation history and a follow-up question,
rephrase the follow-up question to be a standalone question that
contains all the necessary context from the history.

<Conversation History>
{chat_history_str}

<Follow-up Question>
{question}

<Standalone Question>
"""

def format_history(chat_history: List[LlamaChatMessage]) -> str:
    """Helper to format chat history for the rewrite prompt."""
    return "\n".join([f"{msg.role.capitalize()}: {msg.content}" for msg in chat_history])

async def rewrite_query_async(llm: LLM, chat_history: List[LlamaChatMessage], question: str) -> str:
    """Asynchronously rewrites a question using the chat history."""
    if not chat_history:
        return question

    history_str = format_history(chat_history)
    prompt = QUERY_REWRITE_PROMPT_TMPL.format(chat_history_str=history_str, question=question)
    
    response = await llm.acomplete(prompt)
    rewritten_query = response.text.strip()
    print(f"Original query: '{question}' -> Rewritten query: '{rewritten_query}'")
    return rewritten_query

def rewrite_query_sync(llm: LLM, chat_history: List[LlamaChatMessage], question: str) -> str:
    """Synchronously rewrites a question using the chat history."""
    if not chat_history:
        return question

    history_str = format_history(chat_history)
    prompt = QUERY_REWRITE_PROMPT_TMPL.format(chat_history_str=history_str, question=question)

    response = llm.complete(prompt)
    rewritten_query = response.text.strip()
    print(f"Original query: '{question}' -> Rewritten query: '{rewritten_query}'")
    return rewritten_query


# --- 1. Argument Parsing ---
parser = argparse.ArgumentParser(description="Run the RAG system with configurable options.")
parser.add_argument("--chat", action="store_true", help="Run in interactive chat mode in the terminal.")
parser.add_argument("--llm-provider", type=str, default="ollama", choices=["ollama", "gemini"], help="The LLM provider to use.")
parser.add_argument("--model-name", type=str, default="granite3.3:latest", help="The name of the model to use (e.g., 'granite3.3:latest' for ollama, 'models/gemini-1.5-flash' for gemini).")
parser.add_argument("--similarity-top-k", type=int, default=4, help="The number of top similar chunks to retrieve.")
parser.add_argument("--similarity-cutoff", type=float, default=0.7, help="The minimum similarity score for a chunk to be considered (0.0 to 1.0).")
args = parser.parse_args()


# --- 2. Global Settings and Configuration ---
print("--- Starting RAG API Server with Hybrid Chunking Strategy ---")
print("Configuring LlamaIndex settings...")
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
Settings.llm = initialize_llm(args.llm_provider, args.model_name)

semantic_node_parser = SemanticSplitterNodeParser(
    embed_model=Settings.embed_model, 
    breakpoint_percentile_threshold=95
)
print(f"SemanticSplitter is configured and will be used for non-FAQ files.")


# --- 3. Load Documents and Build/Load the Index ---
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

try:
    with open("system_prompt.txt", 'r', encoding='utf-8') as f:
        system_prompt = f.read()
    print("System prompt loaded successfully from system_prompt.txt.")
except FileNotFoundError:
    print("Warning: system_prompt.txt not found. Using a default prompt.", file=sys.stderr)
    system_prompt = "You are a helpful assistant." 

if index:
    retriever = index.as_retriever(
        similarity_top_k=args.similarity_top_k,
        similarity_cutoff=args.similarity_cutoff
    )
    print(f"Retriever configured with top_k={args.similarity_top_k} and cutoff={args.similarity_cutoff}")
else:
    retriever = None

if retriever:
    print("RAG system is ready.")
else:
    print("RAG system could not be initialized.", file=sys.stderr)


# --- 4. Define OpenAI-compatible Pydantic Models (for server mode) ---
class APIChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[APIChatMessage]
    temperature: Optional[float] = 0.1
    stream: Optional[bool] = False

class ChatCompletionResponseChoice(BaseModel):
    index: int = 0
    message: APIChatMessage
    finish_reason: Literal["stop", "length"] = "stop"

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: "chatcmpl-" + str(time.time()))
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "ollama-gemma2-rag"
    choices: List[ChatCompletionResponseChoice]


# --- 5. Create the FastAPI Application (for server mode) ---
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
async def chat_completions(request: ChatCompletionRequest):
    if not retriever:
        raise HTTPException(status_code=503, detail="RAG retriever is not available.")
    
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages found in the request.")

    all_messages = [LlamaChatMessage(role=msg.role, content=msg.content) for msg in request.messages]
    chat_history = all_messages[:-1]
    last_message_content = all_messages[-1].content

    # --- UPDATED: Implement query rewriting flow ---
    # 1. Rewrite the query to be a standalone question
    rewritten_query = await rewrite_query_async(Settings.llm, chat_history, last_message_content)
    
    # 2. Retrieve context using the rewritten query
    retrieved_nodes = await retriever.aretrieve(rewritten_query)
    log_retrieved_chunks(retrieved_nodes, last_message_content, rewritten_query)
    
    # 3. Generate a response using the original history and new context
    context_str = "\n\n".join([n.get_content() for n in retrieved_nodes])
    final_messages_for_llm = [
        LlamaChatMessage(role="system", content=system_prompt),
        *chat_history,
        LlamaChatMessage(role="user", content=f"Given the following context, answer the user's question.\n<Context>\n{context_str}\n</Context>\nUser Question: {last_message_content}")
    ]
    
    response = await Settings.llm.achat(final_messages_for_llm)
    
    assistant_message = APIChatMessage(role="assistant", content=str(response.message.content))
    choice = ChatCompletionResponseChoice(message=assistant_message)
    response_payload = ChatCompletionResponse(choices=[choice])
    
    print(f"[{datetime.now().isoformat()}] Sending response: {response_payload.model_dump_json(indent=2)}")
    return response_payload

# --- 6. Main Execution Block (with mode selection) ---
if __name__ == "__main__":
    if args.chat:
        if not retriever:
            print("Retriever not available. Exiting chat mode.", file=sys.stderr)
            sys.exit(1)
            
        print("\n--- Starting Interactive Chat Mode ---")
        print("Type 'exit' or 'quit' to end the session. Type 'reset' to clear conversation history.")
        
        chat_history = []
        
        while True:
            try:
                user_message = input("\nYou: ")
                if user_message.lower() in ["exit", "quit"]:
                    print("Exiting chat mode. Goodbye!")
                    break
                
                if user_message.lower() == "reset":
                    chat_history = []
                    print("...Conversation history has been reset...")
                    continue

                if not user_message.strip():
                    continue

                print("Bot: Thinking...")
                
                # --- UPDATED: Implement query rewriting flow for interactive mode ---
                # 1. Rewrite query
                rewritten_query = rewrite_query_sync(Settings.llm, chat_history, user_message)
                
                # 2. Retrieve context
                retrieved_nodes = retriever.retrieve(rewritten_query)
                log_retrieved_chunks(retrieved_nodes, user_message, rewritten_query)
                
                # 3. Generate response
                context_str = "\n\n".join([n.get_content() for n in retrieved_nodes])
                final_messages_for_llm = [
                    LlamaChatMessage(role="system", content=system_prompt),
                    *chat_history,
                    LlamaChatMessage(role="user", content=f"Given the following context, answer the user's question.\n<Context>\n{context_str}\n</Context>\nUser Question: {user_message}")
                ]
                
                response = Settings.llm.chat(final_messages_for_llm)
                response_content = response.message.content
                
                print(f"\nBot: {response_content}")
                
                # 4. Update history
                chat_history.append(LlamaChatMessage(role="user", content=user_message))
                chat_history.append(LlamaChatMessage(role="assistant", content=response_content))

            except KeyboardInterrupt:
                print("\nExiting chat mode. Goodbye!")
                break
            except Exception as e:
                print(f"An error occurred: {e}", file=sys.stderr)
    else:
        if not retriever:
            print("Retriever not available. Cannot start server.", file=sys.stderr)
            sys.exit(1)
        print("Starting Uvicorn server at http://localhost:8000")
        print("Access the API documentation at http://localhost:8000/docs")
        print("To inspect chunks, visit http://localhost:8000/chunks")
        uvicorn.run(app, host="0.0.0.0", port=8000)
