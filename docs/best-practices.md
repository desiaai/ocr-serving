# LightOnOCR Usage Patterns & Gotchas

## Why the Gradio Script Works

The script you shared implements **optimal patterns** for LightOnOCR:

### 1. Resolution Management

```python
def render_pdf_page(page, max_resolution=1540, scale=2.77):
    width, height = page.get_size()
    pixel_width = width * scale
    pixel_height = height * scale
    resize_factor = min(1, max_resolution / pixel_width, max_resolution / pixel_height)
    target_scale = scale * resize_factor
    return page.render(scale=target_scale, rev_byteorder=True).to_pil()
```

**Why this matters:**
- `scale=2.77` → 200 DPI rendering (model training resolution)
- `max_resolution=1540` → Recommended by LightOn docs
- Dynamic `resize_factor` → Prevents oversized images while maintaining aspect ratio
- Result: Optimal accuracy vs. payload size tradeoff

### 2. Empty Text Prompt

```python
content = [
    {"type": "text", "text": ""},  # Empty!
    {"type": "image_url", "image_url": {...}}
]
```

**Why empty?**
- Model is **instruction-tuned for OCR by default**
- No prompt = "transcribe everything you see"
- Adding instructions may confuse the model (not trained for Q&A)

### 3. Streaming for UX

```python
payload = {
    "stream": True  # SSE streaming
}

for line in response.iter_lines():
    if line.startswith('data: '):
        chunk = json.loads(line[6:])
        accumulated_response += chunk['choices'][0]['delta']['content']
        yield accumulated_response  # Live Gradio updates
```

**Benefits:**
- User sees text appear in real-time
- No 30-60s blank wait on large documents
- Early detection of errors

## Common Pitfalls

### ❌ Wrong: High-res rendering

```python
# DON'T: 4K rendering
page.render(scale=5.0).to_pil()  # 360 DPI → huge base64 payload
```

**Result:** Timeout, OOM, or API rejection. No accuracy gain beyond 1540px.

### ❌ Wrong: Instruction prompts

```python
# DON'T: Try to control output format
{"type": "text", "text": "Extract text as JSON with bounding boxes"}
```

**Result:** Model trained only for plain text transcription, not structured output.

### ❌ Wrong: Multiple images per request

```python
# DON'T: Batch in single request
content = [
    {"type": "image_url", "image_url": {...}},  # Page 1
    {"type": "image_url", "image_url": {...}}   # Page 2
]
```

**Result:** `--limit-mm-per-prompt '{"image": 1}'` will reject it.

### ✅ Right: Use vLLM batching

```python
# DO: Send separate requests, let vLLM batch internally
async with aiohttp.ClientSession() as session:
    tasks = [process_page(session, page) for page in pages]
    results = await asyncio.gather(*tasks)
```

**Result:** vLLM's `--async-scheduling` batches efficiently.

## Advanced Patterns

### Pattern: Page Range Processing

```python
def extract_pdf_pages(pdf_path, page_range=None):
    """Extract specific pages from PDF"""
    pdf = pdfium.PdfDocument(pdf_path)
    total = len(pdf)

    if page_range:
        start, end = page_range
        pages = range(max(0, start), min(total, end))
    else:
        pages = range(total)

    results = []
    for i in pages:
        img = render_pdf_page(pdf[i])
        text = call_vllm(img)
        results.append({"page": i+1, "text": text})

    pdf.close()
    return results
```

### Pattern: Confidence Detection

```python
# Check for OCR failures
def has_extraction_issues(text):
    """Heuristic checks for bad OCR"""
    if len(text) < 50:  # Too short
        return True
    if text.count('�') > 5:  # Unicode errors
        return True
    if len(set(text)) < 10:  # Low character diversity
        return True
    return False

text = response.json()['choices'][0]['message']['content']
if has_extraction_issues(text):
    print("⚠️ Low confidence extraction, retry with higher resolution?")
```

### Pattern: Table Detection

```python
def likely_contains_table(text):
    """Detect if output is tabular"""
    # LightOn may output markdown tables or tab-separated
    if '|' in text and text.count('|') > 10:
        return True
    if '\t' in text and text.count('\t') > 10:
        return True
    return False

if likely_contains_table(text):
    print("⚠️ Table detected - check raw output")
    print("LightOn table score: 35.2 (weakest category)")
```

## Performance Tuning

### Single Document (latency-optimized)

```bash
vllm serve lightonai/LightOnOCR-1B-1025 \
    --limit-mm-per-prompt '{"image": 1}' \
    --gpu-memory-utilization 0.8 \
    --max-model-len 4096
```

### Batch Processing (throughput-optimized)

```bash
vllm serve lightonai/LightOnOCR-1B-1025 \
    --limit-mm-per-prompt '{"image": 1}' \
    --async-scheduling \
    --max-num-seqs 16 \               # Parallel sequences
    --gpu-memory-utilization 0.95 \
    --max-model-len 4096
```

**Expected throughput:** ~5.71 pages/sec on H100

### Multi-GPU

```bash
vllm serve lightonai/LightOnOCR-1B-1025 \
    --tensor-parallel-size 2 \  # Split across 2 GPUs
    --limit-mm-per-prompt '{"image": 1}' \
    --async-scheduling
```

## Error Handling

```python
def robust_extract(image):
    """Handle common vLLM errors"""
    try:
        response = requests.post(ENDPOINT, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']

    except requests.Timeout:
        print("⚠️ Timeout - image too large?")
        # Retry with lower resolution
        return None

    except requests.HTTPError as e:
        if e.response.status_code == 413:
            print("⚠️ Payload too large")
            return None
        elif e.response.status_code == 503:
            print("⚠️ Server overloaded")
            time.sleep(1)
            return robust_extract(image)  # Retry
        raise

    except KeyError:
        print("⚠️ Unexpected response format")
        return None
```

## Temperature Effects

```python
# OCR tasks
"temperature": 0.0    # Deterministic, best for forms/tables
"temperature": 0.2    # Recommended default (slightly creative)
"temperature": 0.5    # More variation (handwriting?)

# NOT RECOMMENDED
"temperature": 1.0    # Too random for OCR
```

**Rule of thumb:** Lower temp = more reliable character recognition.

## When NOT to Use LightOnOCR

1. **Need structured output** (JSON, XML) → Use GPT-4V + prompt engineering
2. **Complex table parsing** (35.2 score) → Consider Table Transformer or GPT-4V
3. **Non-Latin scripts** (Arabic, Chinese, etc.) → Use tesseract or PaddleOCR
4. **Interactive Q&A** about documents → Use Gemini/Claude with vision
5. **Bounding boxes** needed → Use layout models (LayoutLM, YOLO)

## Best Use Cases

✅ High-volume document digitization (invoices, receipts)
✅ Scientific paper extraction (ArXiv: 81.4 score)
✅ Multi-column layouts (newspapers, magazines)
✅ Math-heavy documents (76.4 score)
✅ Real-time preview UIs (streaming)
✅ Cost-sensitive pipelines (<$0.01/1k pages)

## Integration with Gradio Script

The script you showed hits all optimal patterns:
- ✅ 1540px max resolution
- ✅ Empty text prompt
- ✅ Streaming for UX
- ✅ Temperature=0.2 default
- ✅ Raw + rendered output (handles markdown tables)
- ✅ RGBA→RGB conversion (model expects RGB)

**Only missing:** Error handling for vLLM failures.
