SHELL := /bin/bash

# Auto-detect modal command (prefer installed, fallback to uvx)
MODAL ?= modal
ifeq ($(MODAL),modal)
MODAL_BIN := $(shell command -v modal 2>/dev/null)
ifeq ($(MODAL_BIN),)
MODAL_CMD := uvx modal
else
MODAL_CMD := modal
endif
else
MODAL_CMD := $(MODAL)
endif

# Configuration
MODEL ?= lightonai/LightOnOCR-1B-1025
PDF_URL ?= https://arxiv.org/pdf/2412.13663
PDF_FILE ?=
PAGE ?= 1
MODAL_URL ?=
FAST_BOOT ?= false

# Test PDF options
TEST_PDF := $(if $(PDF_FILE),--pdf-file $(PDF_FILE),--pdf-url $(PDF_URL))

.PHONY: help
help:
	@echo "LightOnOCR Modal Deployment Targets:"
	@echo ""
	@echo "Deployment:"
	@echo "  make serve           - Run Modal dev server (hot reload, temp URL)"
	@echo "  make deploy          - Deploy to Modal (stable URL)"
	@echo "  make serve-uvx       - Run dev server via uvx (no modal install)"
	@echo ""
	@echo "Testing:"
	@echo "  make test            - Test Modal endpoint with arXiv PDF"
	@echo "  make test-custom     - Test with custom PDF (set PDF_FILE=path)"
	@echo "  make health          - Check server health"
	@echo "  make models          - List available models"
	@echo ""
	@echo "Benchmarking:"
	@echo "  make benchmark       - Run throughput benchmark (set MODAL_URL)"
	@echo "  make benchmark-parallel - Test 1, 2, 4 parallel requests"
	@echo "  make metrics         - Show vLLM Prometheus metrics"
	@echo ""
	@echo "Monitoring:"
	@echo "  make logs            - View live logs"
	@echo "  make logs-history    - View historical logs"
	@echo "  make containers      - List running containers"
	@echo ""
	@echo "Configuration:"
	@echo "  MODEL=$(MODEL)"
	@echo "  PDF_URL=$(PDF_URL)"
	@echo "  PAGE=$(PAGE)"
	@echo "  FAST_BOOT=$(FAST_BOOT)"
	@echo "  BENCH_PDF=$(BENCH_PDF)"
	@echo "  BENCH_PAGES=$(BENCH_PAGES)"
	@echo "  BENCH_PARALLEL=$(BENCH_PARALLEL)"
	@echo ""
	@echo "Examples:"
	@echo "  make serve                                    # Start dev server"
	@echo "  make test MODAL_URL=https://...modal.run     # Test endpoint"
	@echo "  make benchmark MODAL_URL=... BENCH_PARALLEL=4 # Benchmark 4 parallel"
	@echo "  FAST_BOOT=true make deploy                   # Deploy with fast boot"

.PHONY: serve
serve:
	@echo "Starting Modal dev server..."
	@echo "Model: $(MODEL)"
	@echo "Fast boot: $(FAST_BOOT)"
	FAST_BOOT=$(FAST_BOOT) MODEL_ID=$(MODEL) $(MODAL_CMD) serve modal_lighton_ocr.py

.PHONY: deploy
deploy:
	@echo "Deploying to Modal..."
	@echo "Model: $(MODEL)"
	@echo "Fast boot: $(FAST_BOOT)"
	FAST_BOOT=$(FAST_BOOT) MODEL_ID=$(MODEL) $(MODAL_CMD) deploy modal_lighton_ocr.py

.PHONY: serve-uvx
serve-uvx:
	@echo "Starting Modal dev server via uvx..."
	FAST_BOOT=$(FAST_BOOT) MODEL_ID=$(MODEL) uvx modal serve modal_lighton_ocr.py

.PHONY: test
test:
	@if [ -z "$(MODAL_URL)" ]; then \
		echo "Error: MODAL_URL not set"; \
		echo "Usage: make test MODAL_URL=https://yourname--lighton-ocr-vllm-serve.modal.run"; \
		exit 1; \
	fi
	@echo "Testing Modal endpoint: $(MODAL_URL)"
	@echo "PDF: $(if $(PDF_FILE),$(PDF_FILE),$(PDF_URL))"
	@echo "Page: $(PAGE)"
	uv run --with pypdfium2 --with pillow --with requests \
	  python tests/test_modal.py \
	  --url $(MODAL_URL) \
	  $(TEST_PDF) \
	  --page $(PAGE) \
	  --model $(MODEL)

.PHONY: test-custom
test-custom:
	@if [ -z "$(MODAL_URL)" ]; then \
		echo "Error: MODAL_URL not set"; \
		exit 1; \
	fi
	@if [ -z "$(PDF_FILE)" ]; then \
		echo "Error: PDF_FILE not set"; \
		echo "Usage: make test-custom MODAL_URL=https://... PDF_FILE=~/invoice.pdf"; \
		exit 1; \
	fi
	@echo "Testing with custom PDF: $(PDF_FILE)"
	uv run --with pypdfium2 --with pillow --with requests \
	  python tests/test_modal.py \
	  --url $(MODAL_URL) \
	  --pdf-file $(PDF_FILE) \
	  --page $(PAGE) \
	  --model $(MODEL)

.PHONY: test-no-stream
test-no-stream:
	@if [ -z "$(MODAL_URL)" ]; then \
		echo "Error: MODAL_URL not set"; \
		exit 1; \
	fi
	uv run --with pypdfium2 --with pillow --with requests \
	  python tests/test_modal.py \
	  --url $(MODAL_URL) \
	  $(TEST_PDF) \
	  --page $(PAGE) \
	  --model $(MODEL) \
	  --no-stream

.PHONY: health
health:
	@if [ -z "$(MODAL_URL)" ]; then \
		echo "Error: MODAL_URL not set"; \
		echo "Usage: make health MODAL_URL=https://yourname--lighton-ocr-vllm-serve.modal.run"; \
		exit 1; \
	fi
	@echo "Health check: $(MODAL_URL)/health"
	@curl -s "$(MODAL_URL)/health" | jq . || curl -s "$(MODAL_URL)/health"

.PHONY: models
models:
	@if [ -z "$(MODAL_URL)" ]; then \
		echo "Error: MODAL_URL not set"; \
		exit 1; \
	fi
	@echo "Available models: $(MODAL_URL)/v1/models"
	@curl -s "$(MODAL_URL)/v1/models" | jq . || curl -s "$(MODAL_URL)/v1/models"

.PHONY: logs
logs:
	@echo "Viewing live logs for lighton-ocr-vllm..."
	$(MODAL_CMD) app logs lighton-ocr-vllm --live

.PHONY: logs-history
logs-history:
	@echo "Viewing historical logs for lighton-ocr-vllm..."
	$(MODAL_CMD) app logs lighton-ocr-vllm

.PHONY: containers
containers:
	@echo "Listing containers for lighton-ocr-vllm..."
	$(MODAL_CMD) container list --app lighton-ocr-vllm

.PHONY: stop
stop:
	@echo "Stopping Modal app..."
	$(MODAL_CMD) app stop lighton-ocr-vllm

.PHONY: install-deps
install-deps:
	@echo "Installing test dependencies..."
	uv pip install pypdfium2 pillow requests

.PHONY: install-modal
install-modal:
	@echo "Installing Modal CLI..."
	uv tool install modal
	@echo ""
	@echo "Now run: modal token new"

.PHONY: check-modal
check-modal:
	@command -v modal >/dev/null 2>&1 && echo "✓ modal CLI installed" || echo "✗ modal CLI not found (run: make install-modal)"
	@command -v uvx >/dev/null 2>&1 && echo "✓ uvx available" || echo "✗ uvx not found"
	@$(MODAL_CMD) token verify >/dev/null 2>&1 && echo "✓ modal authenticated" || echo "✗ not authenticated (run: modal token new)"

.PHONY: clean
clean:
	@echo "Cleaning up __pycache__ and .pyc files..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# Benchmarking
BENCH_PDF ?= /tmp/starbucks.pdf
BENCH_PAGES ?= 1-3
BENCH_PARALLEL ?= 1

.PHONY: benchmark
benchmark:
	@if [ -z "$(MODAL_URL)" ]; then \
		echo "Error: MODAL_URL not set"; \
		exit 1; \
	fi
	@echo "Benchmarking with $(BENCH_PARALLEL) parallel requests..."
	uv run --with pypdfium2 --with pillow --with requests --with aiohttp \
	  python benchmarks/benchmark_modal.py \
	  --url $(MODAL_URL) \
	  --pdf $(BENCH_PDF) \
	  --pages $(BENCH_PAGES) \
	  --parallel $(BENCH_PARALLEL) \
	  --metrics

.PHONY: benchmark-parallel
benchmark-parallel:
	@echo "Testing parallel throughput (1, 2, 4 requests)..."
	@$(MAKE) benchmark BENCH_PARALLEL=1
	@$(MAKE) benchmark BENCH_PARALLEL=2
	@$(MAKE) benchmark BENCH_PARALLEL=4

.PHONY: metrics
metrics:
	@if [ -z "$(MODAL_URL)" ]; then \
		echo "Error: MODAL_URL not set"; \
		exit 1; \
	fi
	@echo "Fetching vLLM metrics from $(MODAL_URL)/metrics"
	@curl -s "$(MODAL_URL)/metrics" | grep -E "vllm:(num_requests|gpu_cache|throughput)" || curl -s "$(MODAL_URL)/metrics"

# Variants for different model sizes
.PHONY: serve-32k
serve-32k:
	@echo "Starting with 32k vocab model (European languages optimized)..."
	$(MAKE) serve MODEL=lightonai/LightOnOCR-1B-32k

.PHONY: deploy-32k
deploy-32k:
	@echo "Deploying with 32k vocab model..."
	$(MAKE) deploy MODEL=lightonai/LightOnOCR-1B-32k

.PHONY: serve-16k
serve-16k:
	@echo "Starting with 16k vocab model (most compact)..."
	$(MAKE) serve MODEL=lightonai/LightOnOCR-1B-16k

.PHONY: deploy-16k
deploy-16k:
	@echo "Deploying with 16k vocab model..."
	$(MAKE) deploy MODEL=lightonai/LightOnOCR-1B-16k

# Fast boot variants
.PHONY: serve-fast
serve-fast:
	@echo "Starting with fast boot enabled (faster cold starts)..."
	$(MAKE) serve FAST_BOOT=true

.PHONY: deploy-fast
deploy-fast:
	@echo "Deploying with fast boot enabled..."
	$(MAKE) deploy FAST_BOOT=true
