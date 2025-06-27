DATA_DIR=data
DATASET_DIR=../acm-observability-llm-ds/dataset

data:
  find .${DATASET_DIR} \( -path '*/processed/*' -o -path '*/faq/*' \) -type f -name "*.md" -exec cp {} ${DATA_DIR} \;

read-chunks:
  cat storage/docstore.json| jq . | less