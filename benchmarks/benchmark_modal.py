#!/usr/bin/env python3
"""
Benchmark script for LightOnOCR Modal deployment
Tests throughput with parallel requests and measures performance

Usage:
    uv run --with pypdfium2 --with pillow --with requests --with aiohttp \
        python benchmark_modal.py \
        --url https://yourname--lighton-ocr-vllm-serve-dev.modal.run \
        --pdf /tmp/starbucks.pdf \
        --pages 1-5 \
        --parallel 4
"""
import argparse
import asyncio
import base64
import io
import json
import time
from pathlib import Path
from typing import List, Dict, Any

try:
    import pypdfium2 as pdfium
    from PIL import Image
    import aiohttp
    import requests
except ImportError as e:
    print(f"Error: {e}")
    print("Run: uv run --with pypdfium2 --with pillow --with requests --with aiohttp python benchmark_modal.py")
    exit(1)


def render_pdf_page(pdf_data: bytes, page_num: int, max_resolution: int = 1540, scale: float = 2.77):
    """Render PDF page to PIL image at LightOnOCR-recommended resolution"""
    pdf = pdfium.PdfDocument(pdf_data)

    if page_num < 1 or page_num > len(pdf):
        raise ValueError(f"Page {page_num} out of range (1-{len(pdf)})")

    page = pdf[page_num - 1]

    # Calculate scaling to keep longest dimension at max_resolution
    width, height = page.get_size()
    pixel_width = width * scale
    pixel_height = height * scale
    resize_factor = min(1, max_resolution / pixel_width, max_resolution / pixel_height)
    target_scale = scale * resize_factor

    pil_image = page.render(scale=target_scale, rev_byteorder=True).to_pil()
    pdf.close()

    return pil_image


def image_to_base64(pil_image):
    """Convert PIL image to base64 string"""
    buffer = io.BytesIO()
    if pil_image.mode == 'RGBA':
        pil_image = pil_image.convert('RGB')
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


async def ocr_page_async(session: aiohttp.ClientSession, url: str, image_base64: str,
                         model_id: str, page_num: int) -> Dict[str, Any]:
    """Run OCR on a single page asynchronously"""
    payload = {
        "model": model_id,
        "messages": [{
            "role": "user",
            "content": [{
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_base64}"}
            }]
        }],
        "max_tokens": 4096,
        "temperature": 0.2,
        "stream": False,  # Non-streaming for benchmarking
    }

    start_time = time.time()

    async with session.post(f"{url}/v1/chat/completions", json=payload) as response:
        response.raise_for_status()
        result = await response.json()

        end_time = time.time()
        duration = end_time - start_time

        text = result['choices'][0]['message']['content']
        tokens = len(text.split())  # Rough token estimate

        return {
            "page": page_num,
            "duration": duration,
            "tokens": tokens,
            "tokens_per_sec": tokens / duration,
            "text_length": len(text),
        }


async def benchmark_parallel(url: str, pdf_data: bytes, pages: List[int],
                             model_id: str, parallel: int, use_semaphore: bool = True) -> Dict[str, Any]:
    """Run parallel OCR requests and measure throughput"""
    print(f"\n{'='*80}")
    print(f"Benchmark: {len(pages)} pages, {parallel} parallel requests")
    print(f"Mode: {'Semaphore (continuous)' if use_semaphore else 'Batch (wait for all)'}")
    print(f"{'='*80}\n")

    # Prepare all images
    print("Preparing images...")
    images = []
    for page_num in pages:
        pil_image = render_pdf_page(pdf_data, page_num)
        image_base64 = image_to_base64(pil_image)
        images.append((page_num, image_base64))
        print(f"  Page {page_num}: {pil_image.size}, {len(image_base64)/1024:.1f}KB base64")

    # Run parallel OCR
    print(f"\nRunning requests...")
    start_time = time.time()

    async with aiohttp.ClientSession() as session:
        if use_semaphore:
            # Semaphore mode: always keep 'parallel' requests in flight
            semaphore = asyncio.Semaphore(parallel)
            all_results = []
            completed = [0]  # Mutable for closure

            async def run_with_semaphore(page_num, img_b64):
                async with semaphore:
                    result = await ocr_page_async(session, url, img_b64, model_id, page_num)
                    completed[0] += 1
                    elapsed = time.time() - start_time
                    print(f"  Completed {completed[0]}/{len(images)} pages in {elapsed:.1f}s "
                          f"(page {page_num}: {result['duration']:.1f}s)")
                    return result

            tasks = [run_with_semaphore(page_num, img_b64) for page_num, img_b64 in images]
            all_results = await asyncio.gather(*tasks)
        else:
            # Batch mode: process in batches of 'parallel' size
            all_results = []
            for i in range(0, len(images), parallel):
                batch = images[i:i+parallel]
                tasks = [
                    ocr_page_async(session, url, img_b64, model_id, page_num)
                    for page_num, img_b64 in batch
                ]
                batch_results = await asyncio.gather(*tasks)
                all_results.extend(batch_results)

                # Print progress
                completed = len(all_results)
                elapsed = time.time() - start_time
                print(f"  Completed {completed}/{len(images)} pages in {elapsed:.1f}s")

    end_time = time.time()
    total_duration = end_time - start_time

    # Calculate statistics
    total_tokens = sum(r["tokens"] for r in all_results)
    avg_duration = sum(r["duration"] for r in all_results) / len(all_results)
    avg_tokens_per_sec = sum(r["tokens_per_sec"] for r in all_results) / len(all_results)

    return {
        "total_pages": len(pages),
        "parallel_requests": parallel,
        "total_duration": total_duration,
        "avg_page_duration": avg_duration,
        "pages_per_sec": len(pages) / total_duration,
        "total_tokens": total_tokens,
        "avg_tokens_per_sec": avg_tokens_per_sec,
        "results": all_results,
    }


def get_metrics(url: str) -> str:
    """Fetch Prometheus metrics from vLLM"""
    response = requests.get(f"{url}/metrics")
    response.raise_for_status()
    return response.text


def parse_key_metrics(metrics_text: str) -> Dict[str, float]:
    """Parse key metrics from Prometheus format"""
    key_metrics = {}

    for line in metrics_text.split('\n'):
        if line.startswith('#') or not line.strip():
            continue

        # Extract metric name and value
        parts = line.split()
        if len(parts) >= 2:
            metric_name = parts[0]
            try:
                metric_value = float(parts[1])

                # Filter for important metrics
                if any(x in metric_name for x in [
                    'vllm:num_requests_running',
                    'vllm:num_requests_waiting',
                    'vllm:gpu_cache_usage',
                    'vllm:avg_prompt_throughput',
                    'vllm:avg_generation_throughput',
                ]):
                    key_metrics[metric_name] = metric_value
            except ValueError:
                pass

    return key_metrics


def main():
    parser = argparse.ArgumentParser(description="Benchmark LightOnOCR Modal deployment")
    parser.add_argument("--url", required=True, help="Modal endpoint URL")
    parser.add_argument("--pdf", required=True, help="PDF file path")
    parser.add_argument("--pages", default="1-3", help="Page range (e.g., 1-5 or 1,3,5)")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel requests")
    parser.add_argument("--model", default="lightonai/LightOnOCR-1B-1025", help="Model ID")
    parser.add_argument("--metrics", action="store_true", help="Show vLLM metrics after benchmark")
    parser.add_argument("--batch", action="store_true", help="Use batch mode instead of semaphore (for testing)")

    args = parser.parse_args()

    # Parse page range
    if '-' in args.pages:
        start, end = map(int, args.pages.split('-'))
        pages = list(range(start, end + 1))
    else:
        pages = [int(p) for p in args.pages.split(',')]

    # Load PDF
    print(f"Loading PDF: {args.pdf}")
    with open(args.pdf, 'rb') as f:
        pdf_data = f.read()

    pdf = pdfium.PdfDocument(pdf_data)
    print(f"PDF has {len(pdf)} pages")
    pdf.close()

    # Run benchmark
    results = asyncio.run(benchmark_parallel(
        args.url, pdf_data, pages, args.model, args.parallel,
        use_semaphore=not args.batch  # Default to semaphore mode
    ))

    # Print results
    print(f"\n{'='*80}")
    print("BENCHMARK RESULTS")
    print(f"{'='*80}\n")
    print(f"Total pages processed:     {results['total_pages']}")
    print(f"Parallel requests:         {results['parallel_requests']}")
    print(f"Total duration:            {results['total_duration']:.2f}s")
    print(f"Average page duration:     {results['avg_page_duration']:.2f}s")
    print(f"Throughput:                {results['pages_per_sec']:.2f} pages/sec")
    print(f"Total tokens extracted:    {results['total_tokens']}")
    print(f"Avg tokens/sec per page:   {results['avg_tokens_per_sec']:.1f}")

    print(f"\nPer-page breakdown:")
    for r in results['results']:
        print(f"  Page {r['page']}: {r['duration']:.2f}s, {r['tokens']} tokens, "
              f"{r['tokens_per_sec']:.1f} tok/s, {r['text_length']} chars")

    # Show metrics
    if args.metrics:
        print(f"\n{'='*80}")
        print("vLLM METRICS")
        print(f"{'='*80}\n")

        metrics_text = get_metrics(args.url)
        key_metrics = parse_key_metrics(metrics_text)

        for name, value in sorted(key_metrics.items()):
            print(f"{name}: {value}")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
