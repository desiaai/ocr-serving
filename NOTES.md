# LightOnOCR-1B-1025 Notes

## Overview

**Specialized OCR vision-language model** optimized for document parsing, not general VLM tasks.

- **Architecture:** Pixtral-based ViT encoder + Qwen3-based text decoder
- **Size:** ~1B parameters (lightweight)
- **License:** Apache 2.0
- **Primary Use:** End-to-end OCR for PDFs, forms, tables, receipts, math notation

## Key Performance Metrics

### Speed Benchmarks
- **5× faster** than dots.ocr
- **2× faster** than PaddleOCR-VL-0.9B
- **1.73× faster** than DeepSeekOCR
- **5.71 pages/sec** on single H100 (~493k pages/day)
- **Cost:** <$0.01 per 1,000 pages

### Accuracy (Olmo-Bench)
| Category | Score |
|----------|-------|
| ArXiv papers | 81.4 |
| Old scans | 71.6 |
| Math notation | 76.4 |
| Tables | 35.2 |
| Multi-column | 80.0 |
| Tiny text | 88.7 |
| Base | 99.5 |
| **Overall** | **76.1** |

## Variants

| Model | Vocab Size | Use Case |
|-------|-----------|----------|
| **LightOnOCR-1B-1025** | 151k | Full multilingual (default) |
| **LightOnOCR-1B-32k** | 32k | European languages (fastest) |
| **LightOnOCR-1B-16k** | 16k | Most compact |

## Critical Implementation Details

### 1. Image Rendering
**Must render PDFs at specific resolution:**
- **Target:** Longest dimension = **1540px**
- **DPI:** 200 (scale factor = 2.77 for 72 DPI PDFs)
- **Format:** PNG or JPEG
- **Maintain aspect ratio** to preserve text geometry

```python
# Correct rendering
page.render(scale=2.77).to_pil()  # 200 DPI
```

### 2. Input Format
Model expects **vision-only input** with optional empty text:

```python
{
  "role": "user",
  "content": [
    {"type": "text", "text": ""},  # Optional/empty
    {
      "type": "image_url",
      "image_url": {"url": "data:image/png;base64,<base64>"}
    }
  ]
}
```

**No prompt engineering needed** - model trained to OCR by default.

### 3. vLLM Server Config
```bash
vllm serve lightonai/LightOnOCR-1B-1025 \
    --limit-mm-per-prompt '{"image": 1}' \
    --async-scheduling
```

**Key flags:**
- `--limit-mm-per-prompt`: One image per request
- `--async-scheduling`: Better throughput for batch processing

### 4. Generation Parameters
```python
{
  "max_tokens": 4096,      # Typical page
  "temperature": 0.2,      # Low for deterministic OCR
  "top_p": 0.9,           # Standard
  "stream": True          # Optional for UI feedback
}
```

## Strengths & Limitations

### Strengths
✅ Fast & cheap for high-volume document processing
✅ End-to-end differentiable (no pipeline dependencies)
✅ Handles complex layouts (multi-column, tables, forms)
✅ Math notation support
✅ Multi-language (EN, FR, DE, ES, IT, NL, PT, SV, DA)

### Limitations
❌ Tables score only 35.2 (weakest category)
❌ Not a general-purpose VLM (no Q&A, no reasoning)
❌ Best for Latin alphabet languages
❌ Requires careful image preprocessing (1540px)

## Training Data
- Scientific papers, books, receipts, invoices, tables, forms
- Handwritten text included
- Real + synthetic document scans
- Multi-language corpus (dataset release pending)

## Future Capabilities
- **LoRA fine-tuning** for domain adaptation
- **Transformers integration** coming soon
- Task-specific corpus fine-tuning (receipts, medical forms, etc.)

## Installation

```bash
# Exact vLLM commit required
export VLLM_COMMIT=e88bdd60d9a25d985168c9f4a60ab10095236d7c

uv pip install vllm \
    'triton-kernels @ git+https://github.com/triton-lang/triton.git@v3.5.0#subdirectory=python/triton_kernels' \
    --torch-backend=auto \
    --extra-index-url https://wheels.vllm.ai/${VLLM_COMMIT} \
    --prerelease=allow

uv pip install pypdfium2 pillow requests
```

**⚠️ Version pinning critical** - uses specific vLLM commit.

## Comparison to General VLMs

| Aspect | LightOnOCR-1B | GPT-4V/Claude | Traditional OCR |
|--------|---------------|---------------|-----------------|
| Speed | 5.71 pg/s | ~0.5 pg/s | Variable |
| Cost | $0.01/1k pg | $10-50/1k pg | Free (local) |
| Tables | ⚠️ Weak | ✅ Strong | ❌ Very weak |
| Math | ✅ Good | ✅ Excellent | ❌ Poor |
| Setup | Single model | API key | Multi-tool pipeline |

## References

- **HuggingFace:** https://huggingface.co/lightonai/LightOnOCR-1B-1025
- **Blog Post:** https://huggingface.co/blog/lightonai/lightonocr/
- **Demo:** https://huggingface.co/spaces/lightonai/LightOnOCR-1B-Demo
- **Colab:** https://colab.research.google.com/#fileId=https%3A//huggingface.co/lightonai/LightOnOCR-1B-1025/blob/main/notebook.ipynb

## Citation

```bibtex
@misc{lightonocr2025,
  title        = {LightOnOCR-1B: End-to-End and Efficient Domain-Specific Vision-Language Models for OCR},
  author       = {Said Taghadouini and Baptiste Aubertin and Adrien Cavaillès},
  year         = {2025},
  howpublished = {\url{https://huggingface.co/blog/lightonai/lightonocr}}
}
```
