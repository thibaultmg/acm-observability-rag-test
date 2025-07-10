# app.py
import os
import sys
import argparse
import yaml
import logging
import uuid
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
    QueryBundle,
)
from llama_index.core.node_parser import SemanticSplitterNodeParser, MarkdownNodeParser
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.postprocessor import SentenceTransformerRerank, SimilarityPostprocessor
from llama_index.core.retrievers import VectorIndexRetriever, QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever


# --- Custom LLM Wrapper for LiteLLM ---
class LiteLLMWrapper(CustomLLM):
    model: str = "ollama/mistral"
    temperature: float = 0.1
    max_tokens: int = 4096

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(context_window=4096, num_output=self.max_tokens, model_name=self.model)

    def _log_token_usage(self, response):
        try:
            usage = response.usage
            logging.info(f"LLM Call ({self.model}) | Input Tokens: {usage.prompt_tokens}, Output Tokens: {usage.completion_tokens}")
        except (AttributeError, KeyError):
            logging.warning(f"Could not log token usage for model {self.model}. Usage data not found in response.")

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        messages = [{"role": "user", "content": prompt}]
        response = litellm.completion(model=self.model, messages=messages, temperature=self.temperature, max_tokens=self.max_tokens)
        self._log_token_usage(response)
        return CompletionResponse(text=response.choices[0].message.content)

    @llm_completion_callback()
    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        messages = [{"role": "user", "content": prompt}]
        response = await litellm.acompletion(model=self.model, messages=messages, temperature=self.temperature, max_tokens=self.max_tokens)
        self._log_token_usage(response)
        return CompletionResponse(text=response.choices[0].message.content)

    @llm_completion_callback()
    def chat(self, messages: Sequence[LlamaChatMessage], **kwargs: Any) -> CompletionResponse:
        litellm_messages = [{"role": msg.role.value, "content": msg.content} for msg in messages]
        response = litellm.completion(model=self.model, messages=litellm_messages, temperature=self.temperature, max_tokens=self.max_tokens)
        self._log_token_usage(response)
        return CompletionResponse(text=response.choices[0].message.content)

    @llm_completion_callback()
    async def achat(self, messages: Sequence[LlamaChatMessage], **kwargs: Any) -> CompletionResponse:
        litellm_messages = [{"role": msg.role.value, "content": msg.content} for msg in messages]
        response = await litellm.acompletion(model=self.model, messages=litellm_messages, temperature=self.temperature, max_tokens=self.max_tokens)
        self._log_token_usage(response)
        return CompletionResponse(text=response.choices[0].message.content)
    
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponse: yield self.complete(prompt, **kwargs)
    def stream_chat(self, messages: Sequence[LlamaChatMessage], **kwargs: Any) -> CompletionResponse: yield self.chat(messages, **kwargs)


# --- Helper Functions ---
def log_retrieved_chunks(source_nodes, query, rewritten_query=None, stage="Retrieved"):
    if logging.getLogger().getEffectiveLevel() > logging.DEBUG: return
    log_message = f"--- {stage} Chunks for Query: {query.strip()} ---\n"
    if rewritten_query: log_message += f"--- Rewritten Query: {rewritten_query.strip()} ---\n"
    log_message += f"{stage} {len(source_nodes)} chunks:\n"
    for i, node_with_score in enumerate(source_nodes):
        node, score = node_with_score.node, node_with_score.score
        log_message += f"  Chunk {i+1} (Score: {score:.4f}) | Source: {node.metadata.get('file_name', 'N/A')}\n"
    logging.debug(log_message)

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

FAITHFULNESS_PROMPT_TMPL = """Given the following context and an answer, you must determine if the answer is fully supported by the context.
The answer must not contain any information that is not explicitly mentioned in the context.
Respond with only the word "YES" if the answer is faithful to the context, and "NO" otherwise.

<Context>
{context_str}
</Context>

<Answer>
{answer}
</Answer>

Is the answer supported by the context?
"""

def format_history(chat_history: List[LlamaChatMessage]) -> str:
    if not chat_history: return "No history yet."
    return "\n".join([f"{msg.role.capitalize()}: {msg.content}" for msg in chat_history])

async def rewrite_query(llm: CustomLLM, chat_history: List[LlamaChatMessage], question: str, product_description: str) -> str:
    history_str = format_history(chat_history)
    prompt = QUERY_REWRITE_PROMPT_TMPL.format(product_description=product_description, chat_history_str=history_str, question=question)
    response = await llm.acomplete(prompt)
    rewritten_query = response.text.strip()
    logging.info(f"Original query: '{question}' -> Rewritten query: '{rewritten_query}'")
    return rewritten_query

def check_faithfulness_sync(llm: CustomLLM, context: str, answer: str) -> bool:
    prompt = FAITHFULNESS_PROMPT_TMPL.format(context_str=context, answer=answer)
    response = llm.complete(prompt)
    result = response.text.strip().upper()
    logging.info(f"Faithfulness check result: {result}")
    return "YES" in result

# --- Configuration Loading ---
def load_config():
    try:
        with open("config.yaml", 'r') as f: config = yaml.safe_load(f)
    except (FileNotFoundError, yaml.YAMLError):
        logging.warning("config.yaml not found or invalid. Using empty config.")
        config = {}
    config.setdefault("server_config", {})
    config.setdefault("llm_config", {})
    config.setdefault("embedding_config", {})
    config.setdefault("retriever_config", {})
    config["embedding_config"]["ollama_host"] = os.getenv("OLLAMA_HOST", config["embedding_config"].get("ollama_host", "http://localhost:11434"))
    return config

# --- Argument Parsing and Logging Setup ---
parser = argparse.ArgumentParser(description="Run the RAG system.")
parser.add_argument("--chat", action="store_true", help="Run in terminal chat mode.")
args, unknown = parser.parse_known_args()

config = load_config()
log_level = config.get("server_config", {}).get("log_level", "INFO").upper()
litellm.logging = False

# --- REMOVED: Session ID logic removed ---
logging.basicConfig(level=getattr(logging, log_level, logging.INFO),
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    stream=sys.stdout)

# --- Chainlit App Logic ---

@cl.on_chat_start
async def on_chat_start():
    logging.info("New chat session started.")
    
    llm_config = config.get("llm_config", {})
    embedding_config = config.get("embedding_config", {})
    retriever_config = config.get("retriever_config", {})

    rewrite_llm = LiteLLMWrapper(model=llm_config.get("rewrite_model", "ollama/mistral"))
    answer_llm = LiteLLMWrapper(model=llm_config.get("answer_model", "ollama/mistral"))
    embed_model = OllamaEmbedding(model_name=embedding_config.get("model_name", "nomic-embed-text"), base_url=embedding_config.get("ollama_host"))
    
    Settings.llm = None
    
    reranker = SentenceTransformerRerank(model="BAAI/bge-reranker-base", top_n=int(retriever_config.get("rerank_top_n", 5)))
    score_filter = SimilarityPostprocessor(similarity_cutoff=float(retriever_config.get("similarity_cutoff", 0.7)))
    logging.info("Reranker and Score Filter initialized.")

    try:
        with open("system_prompt.txt", 'r') as f: system_prompt = f.read()
    except FileNotFoundError: system_prompt = "You are a helpful assistant." 

    try:
        with open("product_description.txt", 'r') as f: product_description = f.read()
    except FileNotFoundError: product_description = "No product description provided."

    DATA_DIR, PERSIST_DIR = "./data", "./storage"
    if os.path.exists(PERSIST_DIR) and os.path.exists(os.path.join(PERSIST_DIR, "docstore.json")):
        logging.info(f"Loading existing index from {PERSIST_DIR}...")
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context, embed_model=embed_model)
    else:
        print("Building new index with Markdown-aware chunking...")
        markdown_parser = MarkdownNodeParser()
        nodes = []
        for filename in os.listdir(DATA_DIR):
            file_path = os.path.join(DATA_DIR, filename)
            if not os.path.isfile(file_path): continue
            if filename.endswith("_faq.md"):
                with open(file_path, 'r', encoding='utf-8') as f: content = f.read()
                chunks = content.split("\n---\n")
                for i, chunk_text in enumerate(chunks):
                    if chunk_text.strip(): nodes.append(TextNode(text=chunk_text.strip(), metadata={"file_name": filename}))
            elif filename.endswith(".md"):
                documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
                nodes.extend(markdown_parser.get_nodes_from_documents(documents))
        
        if nodes:
            index = VectorStoreIndex(nodes, embed_model=embed_model)
            index.storage_context.persist(persist_dir=PERSIST_DIR)
            print("New index built and saved.")
        else:
            await cl.Message(content="No documents found to build index.").send()
            return

    all_nodes = list(index.docstore.docs.values())
    vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=int(retriever_config.get("similarity_top_k", 10)))
    bm25_retriever = BM25Retriever.from_defaults(nodes=all_nodes, similarity_top_k=int(retriever_config.get("similarity_top_k", 10)))
    retriever = QueryFusionRetriever(retrievers=[vector_retriever, bm25_retriever], similarity_top_k=int(retriever_config.get("similarity_top_k", 10)), num_queries=1, mode="reciprocal_rerank", use_async=True, llm=None)
    logging.info("Query Fusion Retriever created.")

    cl.user_session.set("rewrite_llm", rewrite_llm)
    cl.user_session.set("answer_llm", answer_llm)
    cl.user_session.set("retriever", retriever)
    cl.user_session.set("postprocessors", [reranker, score_filter])
    cl.user_session.set("system_prompt", system_prompt)
    cl.user_session.set("product_description", product_description)
    cl.user_session.set("chat_history", [])
    cl.user_session.set("enable_faithfulness_check", llm_config.get("enable_faithfulness_check", False))

    await cl.Message(content="ACM Observability Expert is ready. How can I help you?").send()

@cl.on_message
async def on_message(message: cl.Message):
    rewrite_llm = cl.user_session.get("rewrite_llm")
    answer_llm = cl.user_session.get("answer_llm")
    retriever = cl.user_session.get("retriever")
    postprocessors = cl.user_session.get("postprocessors")
    system_prompt = cl.user_session.get("system_prompt")
    product_description = cl.user_session.get("product_description")
    cl_messages = cl.user_session.get("chat_history")
    enable_faithfulness_check = cl.user_session.get("enable_faithfulness_check")
    
    logging.info(f"Received new message: '{message.content}'")
    llama_history = [LlamaChatMessage(role=m.author, content=m.content) for m in cl_messages]

    msg = cl.Message(content="")
    await msg.send()

    rewritten_query = await rewrite_query(rewrite_llm, llama_history, message.content, product_description)
    
    retrieved_nodes = await retriever.aretrieve(rewritten_query)
    log_retrieved_chunks(retrieved_nodes, message.content, rewritten_query, stage="Fused")
    
    final_nodes = retrieved_nodes
    for postprocessor in postprocessors:
        final_nodes = postprocessor.postprocess_nodes(final_nodes, query_bundle=QueryBundle(rewritten_query))
    log_retrieved_chunks(final_nodes, message.content, rewritten_query, stage="Final")
    
    context_str = "\n\n".join([n.get_content() for n in final_nodes])
    final_messages = [
        LlamaChatMessage(role="system", content=system_prompt),
        *llama_history,
        LlamaChatMessage(role="user", content=f"Context:\n{context_str}\n\nQuestion: {message.content}")
    ]
    
    response = await answer_llm.achat(final_messages)
    final_answer = response.text
    
    if enable_faithfulness_check:
        is_faithful = check_faithfulness_sync(rewrite_llm, context_str, final_answer)
        if not is_faithful:
            final_answer += "\n\n> *Warning: This answer may contain information not present in the source documents.*"

    msg.content = final_answer
    await msg.update()
    logging.info(f"Sent final answer: '{final_answer[:100]}...'")

    cl_messages.append(cl.Message(author="user", content=message.content))
    cl_messages.append(cl.Message(author="assistant", content=final_answer))
    cl.user_session.set("chat_history", cl_messages)

# --- Terminal Chat Mode Logic ---
def run_terminal_chat():
    # ... (Terminal chat logic would need similar updates)
    pass

if __name__ == "__main__":
    if args.chat:
        run_terminal_chat()
    else:
        pass
