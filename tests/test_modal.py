#!/usr/bin/env python3
"""
Test script for LightOnOCR Modal deployment
Downloads an arXiv PDF and sends it to the Modal endpoint for OCR

Dependencies:
    uv pip install pypdfium2 pillow requests
"""
import argparse
import base64
import json
import requests
import io
from pathlib import Path

try:
    import pypdfium2 as pdfium
except ImportError:
    print("Error: pypdfium2 not installed. Run: uv pip install pypdfium2 pillow requests")
    exit(1)

try:
    from PIL import Image
except ImportError:
    print("Error: pillow not installed. Run: uv pip install pypdfium2 pillow requests")
    exit(1)

# Default arXiv paper from LightOnOCR docs
DEFAULT_PDF_URL = "https://arxiv.org/pdf/2412.13663"
DEFAULT_PAGE = 1


def render_pdf_page(pdf_data: bytes, page_num: int = 1, max_resolution: int = 1540, scale: float = 2.77):
    """
    Render a PDF page to PIL image at recommended resolution for LightOnOCR

    Args:
        pdf_data: PDF bytes
        page_num: Page number (1-indexed)
        max_resolution: Max dimension in pixels (1540 recommended by LightOn)
        scale: DPI scale factor (2.77 = 200 DPI for 72 DPI PDFs)
    """
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
    # Convert RGBA to RGB if needed
    if pil_image.mode == 'RGBA':
        pil_image = pil_image.convert('RGB')
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def call_modal_ocr(url: str, image_base64: str, model_id: str, stream: bool = True):
    """
    Call Modal LightOnOCR endpoint

    Args:
        url: Modal endpoint URL
        image_base64: Base64-encoded image
        model_id: Model name
        stream: Whether to stream response
    """
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
        "top_p": 0.9,
        "stream": stream,
    }

    if stream:
        # Streaming response
        response = requests.post(
            f"{url}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            stream=True
        )
        response.raise_for_status()

        accumulated = ""
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    line = line[6:]

                if line.strip() == '[DONE]':
                    break

                try:
                    chunk = json.loads(line)
                    if 'choices' in chunk and len(chunk['choices']) > 0:
                        delta = chunk['choices'][0].get('delta', {})
                        content = delta.get('content', '')
                        if content:
                            print(content, end='', flush=True)
                            accumulated += content
                except json.JSONDecodeError:
                    continue

        print()  # Final newline
        return accumulated
    else:
        # Non-streaming
        response = requests.post(
            f"{url}/v1/chat/completions",
            json=payload
        )
        response.raise_for_status()
        result = response.json()
        text = result['choices'][0]['message']['content']
        print(text)
        return text


def main():
    parser = argparse.ArgumentParser(description="Test LightOnOCR Modal deployment")
    parser.add_argument("--url", required=True, help="Modal endpoint URL (e.g., https://yourname--lighton-ocr-vllm-serve.modal.run)")
    parser.add_argument("--pdf-url", default=DEFAULT_PDF_URL, help="PDF URL to download")
    parser.add_argument("--pdf-file", help="Local PDF file path (alternative to --pdf-url)")
    parser.add_argument("--page", type=int, default=DEFAULT_PAGE, help="Page number (1-indexed)")
    parser.add_argument("--model", default="lightonai/LightOnOCR-1B-1025", help="Model ID")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming")

    args = parser.parse_args()

    # Get PDF data
    if args.pdf_file:
        print(f"Reading local PDF: {args.pdf_file}")
        with open(args.pdf_file, 'rb') as f:
            pdf_data = f.read()
    else:
        print(f"Downloading PDF from: {args.pdf_url}")
        response = requests.get(args.pdf_url)
        response.raise_for_status()
        pdf_data = response.content

    # Render page to image
    print(f"Rendering page {args.page} at 1540px (200 DPI)...")
    pil_image = render_pdf_page(pdf_data, args.page)
    print(f"Image size: {pil_image.size}")

    # Convert to base64
    image_base64 = image_to_base64(pil_image)
    print(f"Base64 payload size: {len(image_base64) / 1024:.1f} KB")

    # Call Modal endpoint
    print(f"\nCalling Modal endpoint: {args.url}")
    print("─" * 80)

    text = call_modal_ocr(
        args.url,
        image_base64,
        args.model,
        stream=not args.no_stream
    )

    print("─" * 80)
    print(f"\nExtracted {len(text)} characters")


if __name__ == "__main__":
    main()
