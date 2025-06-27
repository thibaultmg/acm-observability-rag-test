# chunk_inspector.py
import requests
import json

# --- Configuration ---
# The URL where your RAG server is running
API_BASE_URL = "http://localhost:8000"

def analyze_chunks():
    """
    Fetches chunks from the RAG server's /chunks endpoint and prints statistics.
    """
    print("--- Chunk Inspector ---")
    chunks_url = f"{API_BASE_URL}/chunks"
    
    try:
        print(f"Fetching chunks from {chunks_url}...")
        response = requests.get(chunks_url)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
    except requests.exceptions.RequestException as e:
        print(f"\nError: Could not connect to the RAG server at {API_BASE_URL}.")
        print("Please ensure the server is running and accessible.")
        print(f"Details: {e}")
        return

    chunks = response.json()

    if not isinstance(chunks, list) or not chunks:
        print("No chunks found or unexpected response format.")
        # Print the raw response for debugging
        print("\nServer Response:")
        print(response.text)
        return

    num_chunks = len(chunks)
    char_counts = [chunk['char_count'] for chunk in chunks]
    
    # --- Calculate Statistics ---
    total_chars = sum(char_counts)
    avg_chars = total_chars / num_chunks if num_chunks > 0 else 0
    min_chars = min(char_counts) if num_chunks > 0 else 0
    max_chars = max(char_counts) if num_chunks > 0 else 0

    print("\n--- Chunk Analysis ---")
    print(f"Total number of chunks: {num_chunks}")
    print(f"Average chunk size (characters): {avg_chars:.2f}")
    print(f"Smallest chunk size (characters): {min_chars}")
    print(f"Largest chunk size (characters): {max_chars}")
    print("----------------------\n")

    # --- Display Individual Chunks ---
    print("Inspecting the first 5 chunks:")
    for i, chunk in enumerate(chunks[:5]):
        print(f"\n--- Chunk {i+1} (Node ID: {chunk['node_id']}) ---")
        print(f"Character Count: {chunk['char_count']}")
        print("Text:")
        # Pretty print the text content
        print(json.dumps(chunk['text'], indent=2))
        print("-" * (len(f"--- Chunk {i+1} ---")))

    if num_chunks > 5:
        print(f"\n... and {num_chunks - 5} more chunks.")

if __name__ == "__main__":
    analyze_chunks()
