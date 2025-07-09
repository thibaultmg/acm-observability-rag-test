# ==============================================================================
# Makefile for the RAG Expert System
# ==============================================================================

# --- Configuration ---
# Docker/Podman image settings
QUAY_REGISTRY ?= quay.io/rh-ee-tmange
IMAGE_NAME    ?= acm-observability-expert
TAG           ?= latest
IMAGE         := $(QUAY_REGISTRY)/$(IMAGE_NAME):$(TAG)
OLLAMA_IMAGE  := $(QUAY_REGISTRY)/ollama-nomic-embed-text:latest

# Build configuration
BUILDER       ?= docker
PLATFORMS     ?= linux/amd64,linux/arm64

# Local development settings
DATA_DIR      := ./data
DATASET_DIR   ?= ../acm-observability-llm-ds/dataset

# --- Targets ---

# Default target runs first
.DEFAULT_GOAL := help

# Phony targets are not associated with files
.PHONY: all build push run-compose down-compose run-compose-dev down-compose-dev clean data read-chunks run-ollama run-gemini-chat run-gemini-server help

# ==============================================================================
# Setup & Data Management
# ==============================================================================

data: ## Re-create the data directory from the source dataset
	@echo "Copying dataset files into $(DATA_DIR)..."
	@rm -rf $(DATA_DIR)
	@mkdir -p $(DATA_DIR)
	@find $(DATASET_DIR) \( -path '*/processed/*' -o -path '*/faq/*' \) -type f -name "*.md" -exec cp {} $(DATA_DIR) \;

clean: ## Remove generated files and directories
	@echo "Cleaning up generated files..."
	@rm -rf $(DATA_DIR) storage retrieval.log open-webui-data

# ==============================================================================
# Docker & Container Operations
# ==============================================================================

build: ## Build the container image for the local architecture
	@echo "Building image for local architecture using $(BUILDER)..."
	$(BUILDER) build -t $(IMAGE) .

push: ## Build and push a multi-architecture image to the registry
ifeq ($(BUILDER), podman)
	@echo "Building and pushing multi-arch image with Podman for platforms: $(PLATFORMS)"
	$(BUILDER) build --platform $(PLATFORMS) -t $(IMAGE) .
	$(BUILDER) push $(IMAGE)
else
	@echo "Building and pushing multi-arch image with Docker Buildx for platforms: $(PLATFORMS)"
	docker buildx build --platform $(PLATFORMS) -t $(IMAGE) --push .
endif

.PHONY: build-ollama push-ollama
build-ollama:
	@echo "Building image for local architecture using $(BUILDER)..."
	$(BUILDER) build -f ollama.Dockerfile -t $(OLLAMA_IMAGE) .

push-ollama: ## Build and push a multi-architecture image to the registry
ifeq ($(BUILDER), podman)
	@echo "Building and pushing multi-arch image with Podman for platforms: $(PLATFORMS)"
	$(BUILDER) build -f ollama.Dockerfile --platform $(PLATFORMS) -t $(OLLAMA_IMAGE) .
	$(BUILDER) push $(OLLAMA_IMAGE)
else
	@echo "Building and pushing multi-arch image with Docker Buildx for platforms: $(PLATFORMS)"
	docker buildx build -f ollama.Dockerfile --platform $(PLATFORMS) -t $(OLLAMA_IMAGE) --push .
endif

# ==============================================================================
# Local Development & Testing
# ==============================================================================

run-gemini-server: ## Run the server using Gemini models
	@echo "Starting server with Gemini..."
	@echo "Ensure GOOGLE_API_KEY is set."
	chainlit run main.py

# ==============================================================================
# Help
# ==============================================================================

help: ## Display this help screen
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?#