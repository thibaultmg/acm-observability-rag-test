# app.py
import os
import sys
import argparse
import yaml
from datetime import datetime
from typing import List, Optional, Any, Sequence

import chainlit as cl

# --- LiteLLM and LlamaIndex Imports ---
import litellm
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    LLMMetadata,
    ChatMessage as LlamaChatMessage,
)
from llama_index.core.llms.callbacks import llm_completion_callback
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


# --- Custom LLM Wrapper for LiteLLM ---
class LiteLLMWrapper(CustomLLM):
    model: str = "ollama/mistral"
    temperature: float = 0.1
    max_tokens: int = 2048

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(context_window=4096, num_output=self.max_tokens, model_name=self.model)

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        messages = [{"role": "user", "content": prompt}]
        response = litellm.completion(model=self.model, messages=messages, temperature=self.temperature, max_tokens=self.max_tokens)
        return CompletionResponse(text=response.choices[0].message.content)

    @llm_completion_callback()
    def chat(self, messages: Sequence[LlamaChatMessage], **kwargs: Any) -> CompletionResponse:
        litellm_messages = [{"role": msg.role.value, "content": msg.content} for msg in messages]
        response = litellm.completion(model=self.model, messages=litellm_messages, temperature=self.temperature, max_tokens=self.max_tokens)
        return CompletionResponse(text=response.choices[0].message.content, additional_kwargs={"role": "assistant"})
    
    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse: return self.complete(prompt, **kwargs)
    async def achat(self, messages: Sequence[LlamaChatMessage], **kwargs: Any) -> CompletionResponse: return self.chat(messages, **kwargs)
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponse: yield self.complete(prompt, **kwargs)
    def stream_chat(self, messages: Sequence[LlamaChatMessage], **kwargs: Any) -> CompletionResponse: yield self.chat(messages, **kwargs)


# --- Helper Functions ---
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

FOLLOW_UP_PROMPT_TMPL = """Based on the provided context and the last answer, generate three concise and relevant follow-up questions that a user might ask next.
Return the questions as a numbered list, for example:
1. First question?
2. Second question?
3. Third question?

<Context>
{context_str}
</Context>

<Last Answer>
{answer}
</Last Answer>

<Follow-up Questions>
"""

def format_history(chat_history: List[LlamaChatMessage]) -> str:
    if not chat_history: return "No history yet."
    return "\n".join([f"{msg.role.capitalize()}: {msg.content}" for msg in chat_history])

def rewrite_query_sync(llm: CustomLLM, chat_history: List[LlamaChatMessage], question: str, product_description: str) -> str:
    history_str = format_history(chat_history)
    prompt = QUERY_REWRITE_PROMPT_TMPL.format(product_description=product_description, chat_history_str=history_str, question=question)
    response = llm.complete(prompt)
    rewritten_query = response.text.strip()
    print(f"Original query: '{question}' -> Rewritten query: '{rewritten_query}'")
    return rewritten_query

def generate_follow_ups_sync(llm: CustomLLM, context: str, answer: str) -> List[str]:
    prompt = FOLLOW_UP_PROMPT_TMPL.format(context_str=context, answer=answer)
    response = llm.complete(prompt)
    
    questions = []
    for line in response.text.strip().split('\n'):
        if line.strip():
            parts = line.split('.', 1)
            if len(parts) > 1 and parts[0].isdigit():
                questions.append(parts[1].strip())
            else:
                questions.append(line.strip())
    return questions[:3]

# --- Configuration Loading ---
def load_config():
    try:
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)
    except (FileNotFoundError, yaml.YAMLError):
        print("Warning: config.yaml not found or invalid. Using empty config.", file=sys.stderr)
        config = {}

    config.setdefault("llm_config", {})
    config.setdefault("embedding_config", {})
    config.setdefault("retriever_config", {})
    
    config["embedding_config"]["ollama_host"] = os.getenv(
        "OLLAMA_HOST", config["embedding_config"].get("ollama_host", "http://localhost:11434")
    )
    return config

config = load_config()

# --- Chainlit App Logic ---

@cl.on_chat_start
async def on_chat_start():
    llm_config = config.get("llm_config", {})
    embedding_config = config.get("embedding_config", {})
    retriever_config = config.get("retriever_config", {})

    rewrite_llm = LiteLLMWrapper(model=llm_config.get("rewrite_model", "ollama/mistral"))
    answer_llm = LiteLLMWrapper(model=llm_config.get("answer_model", "ollama/mistral"))
    embed_model = OllamaEmbedding(
        model_name=embedding_config.get("model_name", "nomic-embed-text"),
        base_url=embedding_config.get("ollama_host")
    )
    
    print("--- Initializing RAG System for new session ---")
    print(f"Rewrite LLM: {rewrite_llm.model}")
    print(f"Answer LLM: {answer_llm.model}")
    print(f"Embedding Model: {embed_model.model_name} at {embed_model.base_url}")

    try:
        with open("system_prompt.txt", 'r') as f: system_prompt = f.read()
    except FileNotFoundError: system_prompt = "You are a helpful assistant." 

    try:
        with open("product_description.txt", 'r') as f: product_description = f.read()
    except FileNotFoundError: product_description = "No product description provided."

    DATA_DIR, PERSIST_DIR = "./data", "./storage"
    if os.path.exists(PERSIST_DIR):
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context, embed_model=embed_model)
    else:
        await cl.Message(content="Vector store not found. Please run `make data` and ensure the store is built before starting.").send()
        return

    retriever = index.as_retriever(
        similarity_top_k=int(retriever_config.get("similarity_top_k", 4)),
        similarity_cutoff=float(retriever_config.get("similarity_cutoff", 0.7))
    )

    cl.user_session.set("rewrite_llm", rewrite_llm)
    cl.user_session.set("answer_llm", answer_llm)
    cl.user_session.set("retriever", retriever)
    cl.user_session.set("system_prompt", system_prompt)
    cl.user_session.set("product_description", product_description)
    cl.user_session.set("chat_history", [])

    await cl.Message(content="ACM Observability Expert is ready. How can I help you?").send()

@cl.on_message
async def on_message(message: cl.Message):
    rewrite_llm = cl.user_session.get("rewrite_llm")
    answer_llm = cl.user_session.get("answer_llm")
    retriever = cl.user_session.get("retriever")
    system_prompt = cl.user_session.get("system_prompt")
    product_description = cl.user_session.get("product_description")
    cl_messages = cl.user_session.get("chat_history")
    
    llama_history = [LlamaChatMessage(role=m.author, content=m.content) for m in cl_messages]

    msg = cl.Message(content="")
    await msg.send()

    rewritten_query = rewrite_query_sync(rewrite_llm, llama_history, message.content, product_description)
    retrieved_nodes = retriever.retrieve(rewritten_query)
    log_retrieved_chunks(retrieved_nodes, message.content, rewritten_query)
    
    context_str = "\n\n".join([n.get_content() for n in retrieved_nodes])
    final_messages = [
        LlamaChatMessage(role="system", content=system_prompt),
        *llama_history,
        LlamaChatMessage(role="user", content=f"Context:\n{context_str}\n\nQuestion: {message.content}")
    ]
    
    response = answer_llm.chat(final_messages)
    
    msg.content = response.text
    await msg.update()

    # --- Generate and display follow-up questions ---
    follow_up_questions = generate_follow_ups_sync(rewrite_llm, context_str, response.text)
    if follow_up_questions:
        # --- FIXED: Store the question in the payload for the callback ---
        actions = [
            cl.Action(name="follow_up", value=q, label=q, payload={"content": q}) for q in follow_up_questions
        ]
        await cl.Message(
            content="Here are some suggested follow-ups:",
            actions=actions,
            author="System"
        ).send()

    cl_messages.append(cl.Message(author="user", content=message.content))
    cl_messages.append(cl.Message(author="assistant", content=response.text))
    cl.user_session.set("chat_history", cl_messages)

# --- Add an action callback to handle button clicks ---
@cl.action_callback("follow_up")
async def on_action(action: cl.Action):
    """Handles the 'follow_up' action button clicks."""
    # --- FIXED: Retrieve the question from the action's payload ---
    question_content = action.payload.get("content")
    if question_content:
        await on_message(cl.Message(content=question_content))

# --- Terminal Chat Mode Logic ---
def run_terminal_chat():
    llm_config = config.get("llm_config", {})
    embedding_config = config.get("embedding_config", {})
    retriever_config = config.get("retriever_config", {})

    rewrite_llm = LiteLLMWrapper(model=llm_config.get("rewrite_model"))
    answer_llm = LiteLLMWrapper(model=llm_config.get("answer_model"))
    embed_model = OllamaEmbedding(
        model_name=embedding_config.get("model_name"),
        base_url=embedding_config.get("ollama_host")
    )
    
    try:
        with open("system_prompt.txt", 'r') as f: system_prompt = f.read()
    except FileNotFoundError: system_prompt = "You are a helpful assistant." 

    try:
        with open("product_description.txt", 'r') as f: product_description = f.read()
    except FileNotFoundError: product_description = "No product description provided."

    if not os.path.exists("./storage"):
        print("Vector store not found.", file=sys.stderr); sys.exit(1)
        
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    index = load_index_from_storage(storage_context, embed_model=embed_model)
    retriever = index.as_retriever(
        similarity_top_k=int(retriever_config.get("similarity_top_k", 4)),
        similarity_cutoff=float(retriever_config.get("similarity_cutoff", 0.7))
    )
    
    print("\n--- Starting Interactive Terminal Chat ---")
    chat_history = []
    while True:
        try:
            user_message = input("\nYou: ")
            if user_message.lower() in ["exit", "quit"]: break
            if user_message.lower() == "reset":
                chat_history = []; print("...History reset...")
                continue
            if not user_message.strip(): continue

            print("Bot: Thinking...")
            
            rewritten_query = rewrite_query_sync(rewrite_llm, chat_history, user_message, product_description)
            retrieved_nodes = retriever.retrieve(rewritten_query)
            log_retrieved_chunks(retrieved_nodes, user_message, rewritten_query)
            
            context_str = "\n\n".join([n.get_content() for n in retrieved_nodes])
            final_messages = [
                LlamaChatMessage(role="system", content=system_prompt),
                *chat_history,
                LlamaChatMessage(role="user", content=f"Context:\n{context_str}\n\nQuestion: {user_message}")
            ]
            
            response = answer_llm.chat(final_messages)
            response_content = response.text
            print(f"\nBot: {response_content}")
            
            chat_history.append(LlamaChatMessage(role="user", content=user_message))
            chat_history.append(LlamaChatMessage(role="assistant", content=response_content))

        except KeyboardInterrupt:
            print("\nExiting chat mode."); break
        except Exception as e:
            print(f"An error occurred: {e}", file=sys.stderr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat", action="store_true", help="Run in terminal chat mode.")
    args, unknown = parser.parse_known_args()

    if args.chat:
        run_terminal_chat()
    else:
        # Chainlit will handle running the app in UI mode
        pass
