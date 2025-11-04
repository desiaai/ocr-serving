
---
license: apache-2.0
pipeline_tag: image-to-text
language:
- en
- fr
- de
- es
- it
- nl
- pt
- sv
- da
base_model: lightonai/LightOnOCR-1B-1025
library_name: vllm
tags:
- ocr
- document-understanding
- vision-language
- pdf
- tables
- forms
---

<div align="center">
  <img src="lightonocr-banner.png" alt="LightOn OCR-1B Banner" width="400"/>
</div>

# LightOnOCR-1B-1025

Full BF16 version of the model. We recommend this variant for inference and further fine-tuning.

**LightOnOCR-1B** is a compact, end-to-end vision–language model for Optical Character Recognition (OCR) and document understanding. It achieves state-of-the-art accuracy in its weight class while being several times faster and cheaper than larger general-purpose VLMs.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/#fileId=https%3A//huggingface.co/lightonai/LightOnOCR-1B-1025/blob/main/notebook.ipynb)

📝 **[Read the full blog post](https://huggingface.co/blog/lightonai/lightonocr/)** | 🚀 **[Try the demo](https://huggingface.co/spaces/lightonai/LightOnOCR-1B-Demo)**

**Highlights**

* ⚡ **Speed:** 5× faster than dots.ocr, 2× faster than PaddleOCR-VL-0.9B, 1.73× faster than DeepSeekOCR
* 💸 **Efficiency:** Processes 5.71 pages/s on a single H100 (~493k pages/day) for **<$0.01 per 1,000 pages**
* 🧠 **End-to-End:** Fully differentiable, no external OCR pipeline
* 🧾 **Versatile:** Handles tables, receipts, forms, multi-column layouts, and math notation
* 🌍 **Compact variants:** 32k and 16k vocab options for European languages

---

## Model Overview

**LightOnOCR** combines a Vision Transformer encoder(Pixtral-based) with a lightweight text decoder(Qwen3-based) distilled from high-quality open VLMs.
It is optimized for document parsing tasks, producing accurate, layout-aware text extraction from high-resolution pages.


---

## Benchmarks

| Model              | ArXiv | Old Scans | Math | Tables | Multi-Column | Tiny Text | Base | Overall |
| :----------------- | :---: | :-------: | :--: | :----: | :----------: | :-------: | :--: | :-----: |
| [LightOnOCR-1B-1025](https://huggingface.co/lightonai/LightOnOCR-1B-1025) (151k vocab) | 81.4 | 71.6 | 76.4 | 35.2 | 80.0 | 88.7 | 99.5 | **76.1** |
| [LightOnOCR-1B-32k](https://huggingface.co/lightonai/LightOnOCR-0.9B-32k-1025) (32k vocab) | 80.6 | 66.2 | 73.5 | 33.5 | 71.2 | 87.6 | 99.5 | **73.1** |
| [LightOnOCR-1B-16k](https://huggingface.co/lightonai/LightOnOCR-0.9B-16k-1025) (16k vocab) | 82.3 | 72.9 | 75.3 | 33.5 | 78.6 | 85.1 | 99.8 | **75.4** |

All benchmarks evaluated using **vLLM** on the Olmo-Bench.

---

## Installation

```bash

uv venv --python 3.12 --seed
source .venv/bin/activate

export VLLM_COMMIT=e88bdd60d9a25d985168c9f4a60ab10095236d7c
uv pip install vllm \
    'triton-kernels @ git+https://github.com/triton-lang/triton.git@v3.5.0#subdirectory=python/triton_kernels' \
    --torch-backend=auto \
    --extra-index-url https://wheels.vllm.ai/${VLLM_COMMIT} \
    --prerelease=allow

uv pip install pypdfium2 pillow requests
```

## Start Server

```bash
vllm serve lightonai/LightOnOCR-1B-1025 \
    --limit-mm-per-prompt '{"image": 1}' \
    --async-scheduling
```

## PDF Inference

```python
import base64
import requests
import pypdfium2 as pdfium
import io

ENDPOINT = "http://localhost:8000/v1/chat/completions"
MODEL = "lightonai/LightOnOCR-1B-1025"

# Download PDF from arXiv
pdf_url = "https://arxiv.org/pdf/2412.13663"
pdf_data = requests.get(pdf_url).content

# Open PDF and convert first page to image
pdf = pdfium.PdfDocument(pdf_data)
page = pdf[0]
# Render at 200 DPI (scale factor = 200/72 ≈ 2.77)
pil_image = page.render(scale=2.77).to_pil()

# Convert to base64
buffer = io.BytesIO()
pil_image.save(buffer, format="PNG")
image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

# Make request
payload = {
    "model": MODEL,
    "messages": [{
        "role": "user",
        "content": [{
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_base64}"}
        }]
    }],
    "max_tokens": 4096,
    "temperature": 0.2,
    "top_p": 0.9,
}

response = requests.post(ENDPOINT, json=payload)
text = response.json()['choices'][0]['message']['content']
print(text)
```
---

## Rendering and Preprocessing Tips

* Render PDFs to **PNG** or **JPEG** at a target longest dimension of **1540px**
* Maintain aspect ratio to preserve text geometry
* Use one image per page; batching supported by vLLM

---

## Variants

| Variant                                                                            | Description                                   |
| :--------------------------------------------------------------------------------- | :-------------------------------------------- |
| **[LightOnOCR-1B-1025](https://huggingface.co/lightonai/LightOnOCR-1B-1025)**      | Full multilingual model (default)             |
| **[LightOnOCR-1B-32k](https://huggingface.co/lightonai/LightOnOCR-0.9B-32k-1025)** | Fastest pruned-vocabulary version (32k tokens) optimized for European languages |
| **[LightOnOCR-1B-16k](https://huggingface.co/lightonai/LightOnOCR-0.9B-16k-1025)** | Most compact variant with smallest vocabulary          |

---

## Fine-tuning

**Transformers integration is coming soon for training and inference.**

LightOnOCR is fully differentiable and supports:

* LoRA fine-tuning
* Domain adaptation (receipts, scientific articles, forms, etc.)
* Multilingual fine-tuning with task-specific corpora


Example fine-tuning configurations will be released alongside the dataset.

---

## Data

Trained on a diverse large-scale PDF corpus covering:

* Scientific papers, books, receipts, invoices, tables, forms, and handwritten text
* Multiple languages (Latin alphabet dominant)
* Real and synthetic document scans

The dataset will be released under an open license.

---

## License

Apache License 2.0

---

## Citation

```
@misc{lightonocr2025,
  title        = {LightOnOCR-1B: End-to-End and Efficient Domain-Specific Vision-Language Models for OCR},
  author       = {Said Taghadouini and Baptiste Aubertin and Adrien Cavaillès},
  year         = {2025},
  howpublished = {\url{https://huggingface.co/blog/lightonai/lightonocr}}
}
```