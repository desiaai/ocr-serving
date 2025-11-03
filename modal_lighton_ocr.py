import os
import modal

# Modal vLLM server for LightOnOCR-1B vision model
# Deploys OpenAI-compatible OCR endpoint with vision support

MODEL_ID = "lightonai/LightOnOCR-1B-1025"
N_GPU = 1
VLLM_PORT = 8000
# Default to False for better steady-state throughput (enables compilation + cudagraphs)
FAST_BOOT = os.environ.get("FAST_BOOT", "false").lower() == "true"

# Caches to speed up cold starts
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

# Build-step: download weights into HF cache volume
def preload_model() -> None:
    import os
    from huggingface_hub import snapshot_download

    model_id = os.environ.get("MODEL_ID", MODEL_ID)
    # Download the entire repo snapshot into the standard HF cache path
    snapshot_download(repo_id=model_id)

# Build image with specific vLLM commit required for LightOnOCR
# Per LightOnOCR docs: vLLM commit e88bdd60d9a25d985168c9f4a60ab10095236d7c
# Use pre-built wheels from wheels.vllm.ai (nightlies) until 0.11.1 is released
VLLM_COMMIT = "e88bdd60d9a25d985168c9f4a60ab10095236d7c"

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("git")  # Required for triton-kernels git install
    .uv_pip_install(
        # Follow LightOnOCR's exact install pattern - use pre-built wheels, let deps auto-resolve
        "vllm",
        "triton-kernels @ git+https://github.com/triton-lang/triton.git@v3.5.0#subdirectory=python/triton_kernels",
        "huggingface_hub[hf_transfer]==0.35.0",
        extra_index_url=f"https://wheels.vllm.ai/{VLLM_COMMIT}",
        pre=True,  # Allow prerelease versions
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "MODEL_ID": MODEL_ID,
        "FAST_BOOT": "true" if FAST_BOOT else "false",  # Pass through to container
        "TORCH_CUDA_ARCH_LIST": "89",  # L4 GPU architecture
    })
    .run_function(
        preload_model,
        volumes={"/root/.cache/huggingface": hf_cache_vol},
    )
)

app = modal.App("lighton-ocr-vllm")


@app.function(
    image=image,
    gpu=f"L4:{N_GPU}",
    scaledown_window=15 * 60,  # Stay warm for 15 minutes
    timeout=10 * 60,  # 10 minute timeout
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.concurrent(max_inputs=32)  # Handle 32 concurrent requests
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * 60)
def serve():
    import subprocess
    import os

    model_id = os.environ.get("MODEL_ID", MODEL_ID)

    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        model_id,
        "--served-model-name",
        model_id,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--dtype",
        "auto",
        "--gpu-memory-utilization",
        "0.90",
        "--max-num-seqs",
        "512",
        "--limit-mm-per-prompt",
        '{"image": 1}',  # Vision model: one image per request
        "--async-scheduling",  # Enable async scheduling for better throughput
        "--tensor-parallel-size",
        str(N_GPU),
    ]

    # enforce-eager disables CUDA graphs and Torch compilation
    # Faster cold starts but lower throughput
    cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]

    # Avoid shell=True for proper arg handling
    print("Launching vLLM:", " ".join(cmd))
    subprocess.Popen(cmd)
