# LightOnOCR Modal Deployment Guide

Deploy LightOnOCR-1B vision-language model on Modal with vLLM for fast, scalable OCR.

## Prerequisites

```bash
# Install Modal CLI
uv tool install modal
# or: pip install modal

# Authenticate
modal token new
```

## Quick Start

### 1. Deploy to Modal

**Using Makefile (recommended):**
```bash
cd ~/dev/junk/lighton-vlm

# Check dependencies
make check-modal

# Start dev server (hot reload, temporary URL)
make serve

# Or use uvx without installing modal CLI
make serve-uvx
```

**Direct commands:**
```bash
# Development mode (hot reload, temporary URL)
modal serve modal_lighton_ocr.py

# Production mode (stable URL)
modal deploy modal_lighton_ocr.py
```

This will:
- Build the container image (~5-10 min first time, cached after)
- Pre-download model weights into volume (~500MB)
- Start vLLM server
- Print URL like: `https://yourname--lighton-ocr-vllm-serve-dev.modal.run`

### 2. Test Deployment

**Using Makefile:**
```bash
# Test with arXiv PDF (copy URL from serve/deploy output)
make test MODAL_URL=https://yourname--lighton-ocr-vllm-serve-dev.modal.run

# Test with custom PDF
make test-custom \
  MODAL_URL=https://yourname--lighton-ocr-vllm-serve-dev.modal.run \
  PDF_FILE=~/Documents/invoice.pdf \
  PAGE=2

# Health check
make health MODAL_URL=https://yourname--lighton-ocr-vllm-serve-dev.modal.run

# List models
make models MODAL_URL=https://yourname--lighton-ocr-vllm-serve-dev.modal.run
```

**Direct Python:**
```bash
# Install dependencies
uv pip install pypdfium2 pillow requests

# Test with arXiv PDF
python test_modal.py --url https://yourname--lighton-ocr-vllm-serve-dev.modal.run

# Test with custom PDF
python test_modal.py \
  --url https://yourname--lighton-ocr-vllm-serve-dev.modal.run \
  --pdf-file ~/Documents/invoice.pdf \
  --page 2
```

**With curl:**
```bash
# Health check
curl https://yourname--lighton-ocr-vllm-serve-dev.modal.run/health

# List models
curl https://yourname--lighton-ocr-vllm-serve-dev.modal.run/v1/models
```

## Usage Examples

### Python (OpenAI SDK)

```python
import openai
import base64
import io
from PIL import Image

# Configure client
client = openai.OpenAI(
    base_url="https://yourname--lighton-ocr-vllm-serve.modal.run/v1",
    api_key="not-needed"  # Modal doesn't require API key by default
)

# Load and encode image
image = Image.open("document.png")
buffer = io.BytesIO()
image.save(buffer, format="PNG")
image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

# Call OCR
response = client.chat.completions.create(
    model="lightonai/LightOnOCR-1B-1025",
    messages=[{
        "role": "user",
        "content": [{
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_base64}"}
        }]
    }],
    max_tokens=4096,
    temperature=0.2,
    stream=False
)

print(response.choices[0].message.content)
```

### Python (Streaming)

```python
import requests
import json

response = requests.post(
    "https://yourname--lighton-ocr-vllm-serve.modal.run/v1/chat/completions",
    headers={"Content-Type": "application/json"},
    json={
        "model": "lightonai/LightOnOCR-1B-1025",
        "messages": [{
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}]
        }],
        "stream": True,
        "temperature": 0.2
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        line = line.decode('utf-8')
        if line.startswith('data: ') and line[6:].strip() != '[DONE]':
            chunk = json.loads(line[6:])
            content = chunk['choices'][0]['delta'].get('content', '')
            if content:
                print(content, end='', flush=True)
```

### curl

```bash
# Prepare base64 image
BASE64_IMG=$(base64 -i document.png)

# Call endpoint
curl -X POST https://yourname--lighton-ocr-vllm-serve.modal.run/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"lightonai/LightOnOCR-1B-1025\",
    \"messages\": [{
      \"role\": \"user\",
      \"content\": [{
        \"type\": \"image_url\",
        \"image_url\": {\"url\": \"data:image/png;base64,$BASE64_IMG\"}
      }]
    }],
    \"max_tokens\": 4096,
    \"temperature\": 0.2
  }"
```

## Makefile Commands

All commands support environment variables for configuration:

```bash
# View all available targets
make help

# Common workflows
make serve                          # Start dev server
make deploy                         # Production deploy
make test MODAL_URL=<url>           # Test endpoint
make logs                           # View live logs

# Model variants
make serve-32k                      # 32k vocab (European languages)
make serve-16k                      # 16k vocab (most compact)
make deploy-32k                     # Deploy 32k variant
make deploy-16k                     # Deploy 16k variant

# Fast boot (faster cold starts, lower throughput)
make serve-fast                     # Dev with FAST_BOOT=true
make deploy-fast                    # Deploy with FAST_BOOT=true

# Custom PDF testing
make test-custom \
  MODAL_URL=<url> \
  PDF_FILE=~/invoice.pdf \
  PAGE=2

# Monitoring
make logs                           # Live logs
make logs-history                   # Historical logs
make containers                     # List containers
make health MODAL_URL=<url>         # Health check
make models MODAL_URL=<url>         # List models

# Utilities
make check-modal                    # Check dependencies
make install-modal                  # Install Modal CLI
make clean                          # Remove cache files
```

## Configuration

### Image Preprocessing

**LightOnOCR requires specific image dimensions:**
- **Max dimension:** 1540 pixels (longest side)
- **DPI:** 200 (scale factor 2.77 for standard PDFs)
- **Format:** PNG or JPEG
- **Aspect ratio:** Maintain original

```python
import pypdfium2 as pdfium

def render_pdf_for_lighton(pdf_path, page_num=1):
    """Render PDF at LightOnOCR-recommended resolution"""
    pdf = pdfium.PdfDocument(pdf_path)
    page = pdf[page_num - 1]

    # 2.77 scale = 200 DPI
    # Automatically constrained to 1540px max dimension
    width, height = page.get_size()
    scale = 2.77
    max_resolution = 1540

    pixel_width = width * scale
    pixel_height = height * scale
    resize_factor = min(1, max_resolution / pixel_width, max_resolution / pixel_height)
    target_scale = scale * resize_factor

    return page.render(scale=target_scale).to_pil()
```

### Temperature Settings

```python
temperature=0.0   # Most deterministic (recommended for forms/tables)
temperature=0.2   # Default (slight variation)
temperature=0.5   # More creative (not recommended for OCR)
```

### Environment Variables

Override defaults when deploying:

```bash
# Use different model variant
MODEL_ID=lightonai/LightOnOCR-1B-16k modal deploy modal_lighton_ocr.py

# Enable fast boot (faster cold starts, lower throughput)
FAST_BOOT=true modal deploy modal_lighton_ocr.py
```

## Performance & Cost

### Speed (A100-40GB)
- **Cold start:** ~60-90 seconds (first request after idle)
- **Throughput:** ~1.23 pages/sec (64 parallel requests)
- **Average page latency:** ~38s (under heavy load)

### Cost (A100-40GB)
- **Idle:** $0/hour (scale-to-zero)
- **Active:** $2.10/hour
- **Typical usage (1 hour/day):** ~$63/month
- **Per 1000 pages:** ~$0.47 (at 1.23 pages/sec)

### Scaling Configuration

Edit `modal_lighton_ocr.py`:

```python
@app.function(
    gpu="A100-40GB:1",       # See GPU table for other options
    scaledown_window=15*60,  # Stay warm for 15 min
    timeout=10*60,           # 10 min max
)
@modal.concurrent(max_inputs=64)  # 64 concurrent requests per container
```

**Autoscaling:**
- Scale to zero after 15 minutes of inactivity
- Spin up new containers automatically under load
- Max 64 concurrent requests per container

## Benchmarking

Run throughput benchmarks to measure performance:

```bash
# Basic benchmark
make benchmark \
  MODAL_URL=https://yourname--lighton-ocr-vllm-serve-dev.modal.run \
  BENCH_PDF=/path/to/test.pdf \
  BENCH_PAGES=1-10 \
  BENCH_PARALLEL=4

# High-load benchmark (64 parallel requests)
make benchmark \
  MODAL_URL=https://yourname--lighton-ocr-vllm-serve-dev.modal.run \
  BENCH_PDF=/tmp/document.pdf \
  BENCH_PAGES=1-109 \
  BENCH_PARALLEL=64
```

**Example output (A100-40GB, 109 pages, 64 parallel):**
```
Total pages processed:     109
Parallel requests:         64
Total duration:            88.95s
Average page duration:     38.35s
Throughput:                1.23 pages/sec
Total tokens extracted:    61790
Avg tokens/sec per page:   16.0
```

**Benchmark options:**
- `BENCH_PDF`: Path to PDF file
- `BENCH_PAGES`: Page range (e.g., "1-10" or "1,5,10")
- `BENCH_PARALLEL`: Number of concurrent requests
- `--metrics`: Show vLLM Prometheus metrics after benchmark

## Monitoring

### View Logs
```bash
# Live logs
modal app logs lighton-ocr-vllm --live

# Filter by function
modal app logs lighton-ocr-vllm --function serve
```

### Container Status
```bash
# List running containers
modal container list --app lighton-ocr-vllm

# Execute command in container
modal container exec <container-id> -- vllm --version
```

### Dashboard
View metrics at: https://modal.com/apps

- Request rate
- Latency (p50, p95, p99)
- GPU utilization
- Active containers

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: vllm` | Image build failed. Check logs with `modal app logs lighton-ocr-vllm` |
| `Connection timeout` | Cold start in progress (~60-90s first time). Retry after startup. |
| `413 Payload Too Large` | Image too large. Ensure max dimension is 1540px. |
| `CUDA OOM` | Reduce `--max-num-seqs` in `modal_lighton_ocr.py` |
| `Low accuracy` | Check image resolution (should be ~1540px longest side) |

## Advanced Usage

### GPU Configuration

**Supported GPUs and CUDA Compute Capabilities:**

| GPU Model | CUDA Arch | Memory | Cost/hr | Recommended max-num-seqs |
|-----------|-----------|--------|---------|--------------------------|
| T4 | 75 | 16GB | $0.59 | 256 |
| L4 | 89 | 24GB | $0.80 | 512 |
| A10 | 86 | 24GB | $1.10 | 512 |
| L40S | 89 | 48GB | $1.95 | 2048 |
| A100-40GB | 80 | 40GB | $2.10 | 1024 |
| A100-80GB | 80 | 80GB | $2.50 | 4096 |
| H100 | 90 | 80GB | $3.95 | 4096 |
| H200 | 90 | 141GB | $4.54 | 8192 |
| B200 | 100 | 192GB | $6.25 | 12288 |

**To change GPU in `modal_lighton_ocr.py`:**

```python
# 1. Update CUDA arch in image build:
.env({
    "TORCH_CUDA_ARCH_LIST": "80",  # See CUDA Arch column above
    # ...
})

# 2. Update GPU type:
@app.function(
    gpu="A100-40GB:1",  # GPU Model from table
    # ...
)

# 3. Adjust max sequences in serve():
cmd = [
    # ...
    "--max-num-seqs",
    "1024",  # From Recommended column above
]
```

**CUDA Arch Codes Reference:**
- **100**: B200 (Blackwell)
- **90**: H100, H200 (Hopper)
- **89**: L4, L40S (Ada Lovelace)
- **86**: A10 (Ampere)
- **80**: A100 (Ampere)
- **75**: T4 (Turing)

### Custom Model Variant

Edit `modal_lighton_ocr.py`:

```python
MODEL_ID = "lightonai/LightOnOCR-1B-32k"  # European languages optimized
# or: lightonai/LightOnOCR-1B-16k  # Most compact
```

### Multi-GPU

For larger models or higher throughput:

```python
@app.function(
    gpu="L4:2",  # 2 GPUs
    ...
)
```

Update command:
```python
cmd += ["--tensor-parallel-size", "2"]
```

### Authentication

Add proxy auth for production:

```python
@modal.web_server(port=VLLM_PORT, startup_timeout=10*60, requires_proxy_auth=True)
def serve():
    ...
```

Then call with Modal credentials:
```python
import os
headers = {
    "Modal-Key": os.environ["MODAL_KEY"],
    "Modal-Secret": os.environ["MODAL_SECRET"]
}
```

## API Reference

**Endpoints:**
- `GET /health` - Health check
- `GET /v1/models` - List available models
- `POST /v1/chat/completions` - OCR inference (OpenAI-compatible)

**Request format:**
```json
{
  "model": "lightonai/LightOnOCR-1B-1025",
  "messages": [{
    "role": "user",
    "content": [{
      "type": "image_url",
      "image_url": {"url": "data:image/png;base64,<BASE64>"}
    }]
  }],
  "max_tokens": 4096,
  "temperature": 0.2,
  "stream": true
}
```

**Response format (streaming):**
```
data: {"choices":[{"delta":{"content":"Text"}}]}
data: {"choices":[{"delta":{"content":" here"}}]}
data: [DONE]
```

## Next Steps

1. **Add Gradio UI:** Deploy Gradio frontend as separate Modal app
2. **Batch processing:** Process multiple pages concurrently
3. **Database integration:** Store OCR results in PostgreSQL/S3
4. **Custom domain:** Map to `ocr.yourcompany.com` (Team/Enterprise plans)

## Resources

- **LightOnOCR:** https://huggingface.co/lightonai/LightOnOCR-1B-1025
- **Modal Docs:** https://modal.com/docs
- **vLLM Docs:** https://docs.vllm.ai
- **OpenAI API:** https://platform.openai.com/docs/api-reference
