DATA_DIR=data
DATASET_DIR=../acm-observability-llm-ds/dataset

data:
	find .${DATASET_DIR} \( -path '*/processed/*' -o -path '*/faq/*' \) -type f -name "*.md" -exec cp {} ${DATA_DIR} \;

read-chunks:
	cat storage/docstore.json| jq . | less

run-ollama:
	uv run main.py --chat --llm-provider=ollama --model-name=granite3.3:latest

run-gemini-lite:
	uv run main.py --chat --llm-provider=gemini --model-name=gemini-2.5-flash-lite-preview-06-17

run-gemini-flash:
	uv run main.py --chat --llm-provider=gemini --model-name=gemini-2.5-flash
