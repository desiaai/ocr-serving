# LightOnOCR-1B Modal Deployment

OCR deployment on Modal using LightOnOCR-1B vision-language model via vLLM.

## Features

- One command deployment with Makefile
- OpenAI-compatible API endpoints
- Streaming support for real-time output
- Pre-configured for LightOnOCR (1540px, 200 DPI)
- Scale-to-zero GPU instances

## Quick Start

```bash
# Clone repository
git clone https://github.com/desiaai/ocr-serving.git
cd ocr-serving

# Check dependencies
make check-modal

# Start dev server
FAST_BOOT=false make serve-uvx

# In another terminal, test it (copy URL from serve output)
make test MODAL_URL=https://yourname--lighton-ocr-vllm-serve-dev.modal.run
```

## Project Structure

```
lighton-vlm/
├── modal_lighton_ocr.py       # vLLM server deployment
├── Makefile                   # Deployment automation
├── tests/
│   └── test_modal.py         # Test script
├── benchmarks/
│   └── benchmark_modal.py    # Performance benchmarking
└── docs/
    ├── deployment.md         # Complete usage guide
    ├── best-practices.md     # Usage patterns & gotchas
    └── archive/
        ├── model-card.md     # HuggingFace model card
        └── notes.md          # Model reference
```

## Common Commands

```bash
# Deployment
make serve              # Dev mode (hot reload, temp URL)
make deploy             # Production (stable URL)

# Testing
make test MODAL_URL=<url>                           # Test with arXiv PDF
make test-custom MODAL_URL=<url> PDF_FILE=<path>    # Test custom PDF
make test-custom MODAL_URL=<url> PDF_FILE=<path> PAGE=6  # Test specific page
make health MODAL_URL=<url>                         # Health check

# Monitoring
make logs               # Live logs
make containers         # List containers

# Variants
make serve-32k          # 32k vocab (European languages)
make serve-fast         # Fast cold starts
```

## Architecture

```
┌─────────────────────────────────────┐
│  Modal Container (A100-40GB)        │
│  ┌───────────────────────────────┐  │
│  │  vLLM Server :8000            │  │
│  │  lightonai/LightOnOCR-1B-1025 │  │
│  │  - Vision model                │  │
│  │  - OpenAI-compatible API       │  │
│  │  - Streaming support           │  │
│  └───────────────────────────────┘  │
│                                     │
│  Volumes:                           │
│  - /root/.cache/huggingface (500MB)│
│  - /root/.cache/vllm (CUDA graphs) │
└─────────────────────────────────────┘
```

## Cost Breakdown

A100-40GB pricing: $2.10/hr

| Scenario | Cost/month | Details |
|----------|------------|---------|
| **Idle** | $0 | Scale-to-zero enabled |
| **Light (1 hour/day)** | $63 | ~4% uptime |
| **Medium (4 hours/day)** | $252 | ~17% uptime |
| **24/7** | $1,512 | Always-on (not recommended) |

See [docs/deployment.md](docs/deployment.md) for other GPU options and pricing.

## Model Variants

| Variant | Vocab Size | Best For |
|---------|-----------|----------|
| `LightOnOCR-1B-1025` (default) | 151k | Multilingual (9 languages) |
| `LightOnOCR-1B-32k` | 32k | European languages (fastest) |
| `LightOnOCR-1B-16k` | 16k | Most compact |

```bash
# Use alternative models
make serve MODEL=lightonai/LightOnOCR-1B-32k
```

## Configuration

### FAST_BOOT Toggle

```bash
# Default (FAST_BOOT=false)
make serve
# → 60-90s cold start, higher throughput (CUDA graphs enabled)

# Fast mode (FAST_BOOT=true)
make serve-fast
# → 20-30s cold start, lower throughput (eager execution)
```

### GPU Options

See [docs/deployment.md](docs/deployment.md) for GPU configuration table with CUDA arch codes and pricing.

Current configuration: A100-40GB (CUDA arch 80, $2.10/hr)

## Performance Benchmarks

| Category | LightOnOCR Score | Notes |
|----------|-----------------|-------|
| ArXiv papers | 81.4 | Scientific docs |
| Old scans | 71.6 | Degraded quality |
| Math notation | 76.4 | Equations, symbols |
| **Tables** | **35.2** | Weakest category |
| Multi-column | 80.0 | Newspapers, magazines |
| Tiny text | 88.7 | Small fonts |
| Base | 99.5 | Clean documents |
| **Overall** | **76.1** | Olmo-Bench |

## Limitations

- Tables: 35.2 score (weakest category)
- Not a general VLM (no Q&A, no reasoning)
- Latin alphabet languages only
- Requires 1540px preprocessing

## API Examples

### Python

```python
import requests, json, base64, io
from PIL import Image

# Encode image
img = Image.open("document.png")
buf = io.BytesIO()
img.save(buf, format="PNG")
b64 = base64.b64encode(buf.getvalue()).decode()

# Call API
response = requests.post(
    "https://yourname--lighton-ocr-vllm-serve.modal.run/v1/chat/completions",
    json={
        "model": "lightonai/LightOnOCR-1B-1025",
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
        ]}],
        "temperature": 0.2,
        "stream": True
    },
    stream=True
)

for line in response.iter_lines():
    if line and line.startswith(b'data: '):
        chunk = json.loads(line[6:])
        if chunk.get('choices', [{}])[0].get('delta', {}).get('content'):
            print(chunk['choices'][0]['delta']['content'], end='', flush=True)
```

### curl

```bash
BASE64_IMG=$(base64 -i document.png)

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
    \"temperature\": 0.2
  }"
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `modal: command not found` | Run `make install-modal` |
| Connection timeout on first request | Cold start (60-90s). Wait and retry. |
| `413 Payload Too Large` | Image >1540px. Resize with `pypdfium2`. |
| Low accuracy | Check image is ~1540px and 200 DPI. |
| High cost | Check `make containers` for stuck containers. |

## Documentation

- **[docs/deployment.md](docs/deployment.md)** - Complete deployment guide with examples
- **[docs/best-practices.md](docs/best-practices.md)** - Usage patterns and common pitfalls
- **[docs/archive/model-card.md](docs/archive/model-card.md)** - HuggingFace model card
- **[docs/archive/notes.md](docs/archive/notes.md)** - LightOnOCR model details and benchmarks

## Resources

- **LightOnOCR:** https://huggingface.co/lightonai/LightOnOCR-1B-1025
- **Modal Docs:** https://modal.com/docs
- **vLLM Docs:** https://docs.vllm.ai
- **Blog Post:** https://huggingface.co/blog/lightonai/lightonocr/

## License

LightOnOCR: Apache 2.0

## Next Steps

1. **Deploy:** `make deploy`
2. **Integrate:** Add to your pipeline with OpenAI SDK
3. **Monitor:** Set up alerts in Modal dashboard
4. **Optimize:** Tune `scaledown_window` based on usage patterns
5. **Scale:** Add more GPUs or use model variants
