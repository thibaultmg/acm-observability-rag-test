# main.py
import uvicorn
import time
import os
import sys
import argparse
from datetime import datetime
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Any, Dict, Sequence
import json
import asyncio

# --- NEW: Import LiteLLM and LlamaIndex CustomLLM components ---
import litellm
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    ChatMessage as LlamaChatMessage,
)
from llama_index.core.llms.callbacks import llm_completion_callback

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
from llama_index.embeddings.ollama import OllamaEmbedding


# --- NEW: Custom LLM Wrapper for LiteLLM ---
class LiteLLMWrapper(CustomLLM):
    model: str = "ollama/mistral"
    temperature: float = 0.1
    max_tokens: int = 2048

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=4096,  # A generic default, adjust if needed
            num_output=self.max_tokens,
            model_name=self.model,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Synchronous completion method."""
        messages = [{"role": "user", "content": prompt}]
        response = litellm.completion(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        text = response.choices[0].message.content
        return CompletionResponse(text=text)

    @llm_completion_callback()
    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Asynchronous completion method."""
        messages = [{"role": "user", "content": prompt}]
        response = await litellm.acompletion(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        text = response.choices[0].message.content
        return CompletionResponse(text=text)

    @llm_completion_callback()
    def chat(self, messages: Sequence[LlamaChatMessage], **kwargs: Any) -> CompletionResponse:
        """Synchronous chat method."""
        # Convert LlamaIndex messages to LiteLLM format
        litellm_messages = [{"role": msg.role.value, "content": msg.content} for msg in messages]
        response = litellm.completion(
            model=self.model,
            messages=litellm_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        text = response.choices[0].message.content
        return CompletionResponse(text=text, additional_kwargs={"role": "assistant"})

    @llm_completion_callback()
    async def achat(self, messages: Sequence[LlamaChatMessage], **kwargs: Any) -> CompletionResponse:
        """Asynchronous chat method."""
        litellm_messages = [{"role": msg.role.value, "content": msg.content} for msg in messages]
        response = await litellm.acompletion(
            model=self.model,
            messages=litellm_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        # Accessing message content correctly from the async response
        text = response.choices[0].message.content
        return CompletionResponse(text=text, additional_kwargs={"role": "assistant"})

    # Stream methods are not implemented for this example
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        raise NotImplementedError("Streaming not implemented.")

    def stream_chat(self, messages: Sequence[LlamaChatMessage], **kwargs: Any) -> CompletionResponseGen:
        raise NotImplementedError("Streaming not implemented.")


def log_retrieved_chunks(source_nodes, query, rewritten_query=None):
    if not source_nodes: return
    with open("retrieval.log", "a", encoding="utf-8") as f:
        log_entry = f"--- Original Query: {query.strip()} ---\n"
        if rewritten_query: log_entry += f"--- Rewritten Query: {rewritten_query.strip()} ---\n"
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

QUERY_REWRITE_PROMPT_TMPL = """You are an expert assistant who rewrites a user's question to be a standalone question.
Your goal is to rephrase the question to include all necessary context from the chat history and product description.
Use precise terminology from the product description to optimize retrieval in a RAG system.
Do not answer the question. Only provide the rephrased, standalone question.

<Product Description>
{product_description}

<Conversation History>
{chat_history_str}

<User Question>
{question}

<Standalone Question>
"""

def format_history(chat_history: List[LlamaChatMessage]) -> str:
    if not chat_history: return "No history yet."
    return "\n".join([f"{msg.role.capitalize()}: {msg.content}" for msg in chat_history])

async def rewrite_query_async(llm: CustomLLM, chat_history: List[LlamaChatMessage], question: str, product_description: str) -> str:
    history_str = format_history(chat_history)
    prompt = QUERY_REWRITE_PROMPT_TMPL.format(product_description=product_description, chat_history_str=history_str, question=question)
    response = await llm.acomplete(prompt)
    rewritten_query = response.text.strip()
    print(f"Original query: '{question}' -> Rewritten query: '{rewritten_query}'")
    return rewritten_query

def rewrite_query_sync(llm: CustomLLM, chat_history: List[LlamaChatMessage], question: str, product_description: str) -> str:
    history_str = format_history(chat_history)
    prompt = QUERY_REWRITE_PROMPT_TMPL.format(product_description=product_description, chat_history_str=history_str, question=question)
    response = llm.complete(prompt)
    rewritten_query = response.text.strip()
    print(f"Original query: '{question}' -> Rewritten query: '{rewritten_query}'")
    return rewritten_query

# --- 1. Argument Parsing ---
parser = argparse.ArgumentParser(description="Run the RAG system with configurable options.")
parser.add_argument("--chat", action="store_true", help="Run in interactive chat mode in the terminal.")
# --- UPDATED: Arguments for LiteLLM models ---
parser.add_argument("--rewrite-model", type=str, default="ollama/granite-3b-code-instruct", help="The LiteLLM model name for query rewriting (e.g., 'ollama/llama3').")
parser.add_argument("--answer-model", type=str, default="ollama/granite3.3:latest", help="The LiteLLM model name for final answering (e.g., 'gemini/gemini-1.5-pro').")
parser.add_argument("--similarity-top-k", type=int, default=4, help="The number of top similar chunks to retrieve.")
parser.add_argument("--similarity-cutoff", type=float, default=0.7, help="The minimum similarity score for a chunk to be considered (0.0 to 1.0).")
args = parser.parse_args()

# --- 2. Global Settings and Configuration ---
print("--- Starting RAG API Server with LiteLLM Integration ---")
print("Configuring LlamaIndex settings...")
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# --- UPDATED: Initialize separate LLMs for rewriting and answering ---
rewrite_llm = LiteLLMWrapper(model=args.rewrite_model)
answer_llm = LiteLLMWrapper(model=args.answer_model)
Settings.llm = answer_llm # Set the default LLM to the answer model

print(f"Using '{args.rewrite_model}' for query rewriting.")
print(f"Using '{args.answer_model}' for final answering.")

semantic_node_parser = SemanticSplitterNodeParser(embed_model=Settings.embed_model, breakpoint_percentile_threshold=95)
print(f"SemanticSplitter is configured and will be used for non-FAQ files.")

# ... (rest of the script remains largely the same, but with updated LLM calls) ...

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
            if not os.path.isfile(file_path): continue
            if filename.endswith("_faq.md"):
                print(f"Applying custom '---' splitter for: {filename}")
                with open(file_path, 'r', encoding='utf-8') as f: content = f.read()
                chunks = content.split("\n---\n")
                for i, chunk_text in enumerate(chunks):
                    if chunk_text.strip():
                        nodes.append(TextNode(text=chunk_text.strip(), metadata={"file_name": filename, "chunk_number": i}))
            else:
                print(f"Applying semantic splitter for: {filename}")
                documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
                nodes.extend(semantic_node_parser.get_nodes_from_documents(documents))
        if not nodes:
            print("No documents found.")
            index = None
        else:
            index = VectorStoreIndex(nodes)
            index.storage_context.persist(persist_dir=PERSIST_DIR)
            print(f"New index built and saved to {PERSIST_DIR}.")
    except Exception as e:
        print(f"Error loading documents or building index: {e}", file=sys.stderr)
        sys.exit(1)

try:
    with open("system_prompt.txt", 'r', encoding='utf-8') as f: system_prompt = f.read()
    print("System prompt loaded successfully from system_prompt.txt.")
except FileNotFoundError:
    print("Warning: system_prompt.txt not found.", file=sys.stderr)
    system_prompt = "You are a helpful assistant." 

try:
    with open("product_description.txt", 'r', encoding='utf-8') as f: product_description = f.read()
    print("Product description loaded successfully from product_description.txt.")
except FileNotFoundError:
    print("Warning: product_description.txt not found.", file=sys.stderr)
    product_description = "No product description provided."

if index:
    retriever = index.as_retriever(similarity_top_k=args.similarity_top_k, similarity_cutoff=args.similarity_cutoff)
    print(f"Retriever configured with top_k={args.similarity_top_k} and cutoff={args.similarity_cutoff}")
else:
    retriever = None

if retriever: print("RAG system is ready.")
else: print("RAG system could not be initialized.", file=sys.stderr)

# --- 4. Define OpenAI-compatible Pydantic Models ---
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
    model: str = "litellm-rag"
    choices: List[ChatCompletionResponseChoice]

# --- 5. Create the FastAPI Application ---
app = FastAPI()

@app.get("/health", summary="Health Check")
def health_check(): return {"status": "ok"}

@app.get("/chunks", summary="Inspect Document Chunks")
def get_chunks():
    if not index: raise HTTPException(status_code=503, detail="Index is not available.")
    nodes = index.docstore.docs.values()
    if not nodes: return {"message": "No nodes found in the index."}
    chunks_data = [{"node_id": n.node_id, "text": n.get_content(), "metadata": n.metadata, "char_count": len(n.get_content())} for n in nodes]
    return Response(content=json.dumps(chunks_data, indent=2), media_type="application/json")

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse, summary="OpenAI-compatible Chat Endpoint")
async def chat_completions(request: ChatCompletionRequest):
    if not retriever: raise HTTPException(status_code=503, detail="RAG retriever is not available.")
    if not request.messages: raise HTTPException(status_code=400, detail="No messages found.")
    
    all_messages = [LlamaChatMessage(role=msg.role, content=msg.content) for msg in request.messages]
    chat_history = all_messages[:-1]
    last_message_content = all_messages[-1].content

    # 1. Rewrite query using the dedicated rewrite_llm
    rewritten_query = await rewrite_query_async(rewrite_llm, chat_history, last_message_content, product_description)
    
    # 2. Retrieve context
    retrieved_nodes = await retriever.aretrieve(rewritten_query)
    log_retrieved_chunks(retrieved_nodes, last_message_content, rewritten_query)
    
    # 3. Generate response using the dedicated answer_llm
    context_str = "\n\n".join([n.get_content() for n in retrieved_nodes])
    final_messages_for_llm = [
        LlamaChatMessage(role="system", content=system_prompt),
        *chat_history,
        LlamaChatMessage(role="user", content=f"Given the following context, answer the user's question.\n<Context>\n{context_str}\n</Context>\nUser Question: {last_message_content}")
    ]
    
    response = await answer_llm.achat(final_messages_for_llm)
    
    assistant_message = APIChatMessage(role="assistant", content=str(response.text))
    choice = ChatCompletionResponseChoice(message=assistant_message)
    response_payload = ChatCompletionResponse(choices=[choice])
    
    print(f"[{datetime.now().isoformat()}] Sending response: {response_payload.model_dump_json(indent=2)}")
    return response_payload

# --- 6. Main Execution Block ---
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
                if user_message.lower() in ["exit", "quit"]: break
                if user_message.lower() == "reset":
                    chat_history = []
                    print("...Conversation history has been reset...")
                    continue
                if not user_message.strip(): continue

                print("Bot: Thinking...")
                
                # 1. Rewrite query with rewrite_llm
                rewritten_query = rewrite_query_sync(rewrite_llm, chat_history, user_message, product_description)
                
                # 2. Retrieve context
                retrieved_nodes = retriever.retrieve(rewritten_query)
                log_retrieved_chunks(retrieved_nodes, user_message, rewritten_query)
                
                # 3. Generate response with answer_llm
                context_str = "\n\n".join([n.get_content() for n in retrieved_nodes])
                final_messages_for_llm = [
                    LlamaChatMessage(role="system", content=system_prompt),
                    *chat_history,
                    LlamaChatMessage(role="user", content=f"Given the following context, answer the user's question.\n<Context>\n{context_str}\n</Context>\nUser Question: {user_message}")
                ]
                
                response = answer_llm.chat(final_messages_for_llm)
                response_content = response.text
                
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
