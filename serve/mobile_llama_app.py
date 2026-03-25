"""
Falcon H1 Mobile Model Serving via llama-cpp-python + Ray Serve

Serves all 16 Falcon H1 GGUF models from mobile_model.py behind an
OpenAI-compatible chat API.  Mobile clients select a model by its
"value" identifier (e.g. "h1-tiny-90m-q4", "h1-1.5b-q4").

Designed for 3 x L4 GPUs (24 GB each).
GGUF files are downloaded from HuggingFace at deployment startup.
"""

import json
import logging
import os
from typing import Dict

from ray import serve
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mobile_llm")

# Read from env var injected by the deployment (FALCON_USE_GPU=1 for GPU, 0 for CPU).
USE_GPU: bool = os.environ.get("FALCON_USE_GPU", "1") == "1"

# ---------------------------------------------------------------------------
# System prompts (from mobile_model.py)
# ---------------------------------------------------------------------------
_SYS_TINY = "You are Falcon, a helpful assistant."
_SYS_SMALL = "You are Falcon, a helpful AI assistant by TII. Keep responses concise."
_SYS_MEDIUM = (
    "You are Falcon, a helpful AI assistant by TII. "
    "Keep responses concise and to the point. Use short paragraphs."
)

# ---------------------------------------------------------------------------
# Full model catalogue – mirrors mobile_model.py exactly.
# ---------------------------------------------------------------------------
MODEL_CONFIGS: Dict[str, dict] = {
    # ── Falcon H1 Tiny 90M ──────────────────────────────────────────
    "h1-tiny-90m-q4": {
        "repo_id": "tiiuae/Falcon-H1-Tiny-90M-Instruct-GGUF",
        "filename": "Falcon-H1-Tiny-90M-Instruct-Q4_K_M.gguf",
        "ctx": 512, "max_tokens": 256,
        "temperature": 0.2, "top_p": 0.85, "top_k": 20,
        "system_prompt": _SYS_TINY,
        "num_gpus": 0.04,
    },
    "h1-tiny-90m-q8": {
        "repo_id": "tiiuae/Falcon-H1-Tiny-90M-Instruct-GGUF",
        "filename": "Falcon-H1-Tiny-90M-Instruct-Q8_0.gguf",
        "ctx": 512, "max_tokens": 256,
        "temperature": 0.2, "top_p": 0.85, "top_k": 20,
        "system_prompt": _SYS_TINY,
        "num_gpus": 0.04,
    },
    "h1-tiny-90m-f16": {
        "repo_id": "tiiuae/Falcon-H1-Tiny-90M-Instruct-GGUF",
        "filename": "Falcon-H1-Tiny-90M-Instruct-BF16.gguf",
        "ctx": 512, "max_tokens": 256,
        "temperature": 0.3, "top_p": 0.85, "top_k": 20,
        "system_prompt": _SYS_TINY,
        "num_gpus": 0.05,
    },
    # ── Falcon H1 Tiny 100M Multilingual ─────────────────────────────
    "h1-tiny-100m-q4": {
        "repo_id": "mradermacher/Falcon-H1-Tiny-Multilingual-100M-Instruct-GGUF",
        "filename": "Falcon-H1-Tiny-Multilingual-100M-Instruct.Q4_K_M.gguf",
        "ctx": 512, "max_tokens": 256,
        "temperature": 0.2, "top_p": 0.85, "top_k": 20,
        "system_prompt": _SYS_TINY,
        "num_gpus": 0.04,
    },
    "h1-tiny-100m-q8": {
        "repo_id": "mradermacher/Falcon-H1-Tiny-Multilingual-100M-Instruct-GGUF",
        "filename": "Falcon-H1-Tiny-Multilingual-100M-Instruct.Q8_0.gguf",
        "ctx": 512, "max_tokens": 256,
        "temperature": 0.2, "top_p": 0.85, "top_k": 20,
        "system_prompt": _SYS_TINY,
        "num_gpus": 0.04,
    },
    "h1-tiny-100m-f16": {
        "repo_id": "mradermacher/Falcon-H1-Tiny-Multilingual-100M-Instruct-GGUF",
        "filename": "Falcon-H1-Tiny-Multilingual-100M-Instruct.f16.gguf",
        "ctx": 512, "max_tokens": 256,
        "temperature": 0.3, "top_p": 0.85, "top_k": 20,
        "system_prompt": _SYS_TINY,
        "num_gpus": 0.05,
    },
    # ── Falcon H1 Tiny R 0.6B ───────────────────────────────────────
    "h1-tiny-0.6b-q4": {
        "repo_id": "tiiuae/Falcon-H1-Tiny-R-0.6B-GGUF",
        "filename": "Falcon-H1R-0.6B-Q4_K_M.gguf",
        "ctx": 1024, "max_tokens": 512,
        "temperature": 0.3, "top_p": 0.88, "top_k": 30,
        "system_prompt": _SYS_SMALL,
        "num_gpus": 0.08,
    },
    "h1-tiny-0.6b-q8": {
        "repo_id": "tiiuae/Falcon-H1-Tiny-R-0.6B-GGUF",
        "filename": "Falcon-H1R-0.6B-Q8_0.gguf",
        "ctx": 1024, "max_tokens": 512,
        "temperature": 0.4, "top_p": 0.88, "top_k": 30,
        "system_prompt": _SYS_SMALL,
        "num_gpus": 0.08,
    },
    "h1-tiny-0.6b-f16": {
        "repo_id": "tiiuae/Falcon-H1-Tiny-R-0.6B-GGUF",
        "filename": "Falcon-H1R-0.6B-BF16.gguf",
        "ctx": 1024, "max_tokens": 512,
        "temperature": 0.4, "top_p": 0.88, "top_k": 30,
        "system_prompt": _SYS_SMALL,
        "num_gpus": 0.10,
    },
    # ── Falcon H1 0.5B ──────────────────────────────────────────────
    "h1-0.5b-q4": {
        "repo_id": "unsloth/Falcon-H1-0.5B-Instruct-GGUF",
        "filename": "Falcon-H1-0.5B-Instruct-UD-Q4_K_XL.gguf",
        "ctx": 1024, "max_tokens": 512,
        "temperature": 0.3, "top_p": 0.88, "top_k": 30,
        "system_prompt": _SYS_SMALL,
        "num_gpus": 0.08,
    },
    "h1-0.5b-q8": {
        "repo_id": "unsloth/Falcon-H1-0.5B-Instruct-GGUF",
        "filename": "Falcon-H1-0.5B-Instruct-UD-Q8_K_XL.gguf",
        "ctx": 1024, "max_tokens": 512,
        "temperature": 0.4, "top_p": 0.88, "top_k": 30,
        "system_prompt": _SYS_SMALL,
        "num_gpus": 0.08,
    },
    "h1-0.5b-f16": {
        "repo_id": "unsloth/Falcon-H1-0.5B-Instruct-GGUF",
        "filename": "Falcon-H1-0.5B-Instruct-BF16.gguf",
        "ctx": 1024, "max_tokens": 512,
        "temperature": 0.4, "top_p": 0.88, "top_k": 30,
        "system_prompt": _SYS_SMALL,
        "num_gpus": 0.10,
    },
    # ── Falcon H1 1.5B (recommended) ───────────────────────────
    "h1-1.5b-q4": {
        "repo_id": "unsloth/Falcon-H1-1.5B-Deep-Instruct-GGUF",
        "filename": "Falcon-H1-1.5B-Deep-Instruct-UD-Q4_K_XL.gguf",
        "ctx": 2048, "max_tokens": 1024,
        "temperature": 0.4, "top_p": 0.9, "top_k": 35,
        "system_prompt": _SYS_MEDIUM,
        "num_gpus": 0.15,
    },
    "h1-1.5b-q8": {
        "repo_id": "unsloth/Falcon-H1-1.5B-Deep-Instruct-GGUF",
        "filename": "Falcon-H1-1.5B-Deep-Instruct-UD-Q8_K_XL.gguf",
        "ctx": 2048, "max_tokens": 1024,
        "temperature": 0.5, "top_p": 0.9, "top_k": 35,
        "system_prompt": _SYS_MEDIUM,
        "num_gpus": 0.15,
    },
    # ── Falcon H1 3B ────────────────────────────────────────────────
    "h1-3b-q4": {
        "repo_id": "unsloth/Falcon-H1-3B-Instruct-GGUF",
        "filename": "Falcon-H1-3B-Instruct-UD-Q4_K_XL.gguf",
        "ctx": 1024, "max_tokens": 1024,
        "temperature": 0.5, "top_p": 0.92, "top_k": 40,
        "system_prompt": _SYS_MEDIUM,
        "num_gpus": 0.20,
    },
}


# ---------------------------------------------------------------------------
# llama.cpp model deployment
# ---------------------------------------------------------------------------
@serve.deployment(
    max_ongoing_requests=20,
    health_check_period_s=30,
    health_check_timeout_s=300,
)
class FalconH1Model:
    def __init__(self, model_name: str, config: dict):
        self.model_name = model_name
        self.config = config

        from huggingface_hub import hf_hub_download
        from llama_cpp import Llama

        logger.info(
            f"[{model_name}] Downloading {config['repo_id']}/{config['filename']} ..."
        )
        gguf_path = hf_hub_download(
            repo_id=config["repo_id"],
            filename=config["filename"],
        )
        logger.info(f"[{model_name}] GGUF cached at {gguf_path}")

        logger.info(f"[{model_name}] Loading model (n_ctx={config['ctx']}) ...")
        n_gpu_layers = -1 if USE_GPU else 0   # -1 = all layers on GPU; 0 = CPU only
        self.llm = Llama(
            model_path=gguf_path,
            n_ctx=config["ctx"],
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        logger.info(f"[{model_name}] Model loaded and ready")

    def _prepare_messages(self, messages):
        sys_prompt = self.config.get("system_prompt", "")
        if sys_prompt and (not messages or messages[0].get("role") != "system"):
            messages = [{"role": "system", "content": sys_prompt}] + messages
        return messages

    async def generate(self, request_dict: dict) -> dict:
        messages = self._prepare_messages(request_dict.get("messages", []))
        model_alias = request_dict.get("model", self.model_name)
        max_tokens = min(
            int(request_dict.get("max_tokens", self.config["max_tokens"])),
            4096,
        )
        temperature = float(
            request_dict.get("temperature", self.config["temperature"])
        )
        top_p = float(request_dict.get("top_p", self.config["top_p"]))
        top_k = int(request_dict.get("top_k", self.config["top_k"]))

        result = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stream=False,
        )

        # Overwrite the model name with the alias the client sent
        result["model"] = model_alias
        return result

    async def generate_stream(self, request_dict: dict):
        messages = self._prepare_messages(request_dict.get("messages", []))
        model_alias = request_dict.get("model", self.model_name)
        max_tokens = min(
            int(request_dict.get("max_tokens", self.config["max_tokens"])),
            4096,
        )
        temperature = float(
            request_dict.get("temperature", self.config["temperature"])
        )
        top_p = float(request_dict.get("top_p", self.config["top_p"]))
        top_k = int(request_dict.get("top_k", self.config["top_k"]))

        stream = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stream=True,
        )

        for chunk in stream:
            chunk["model"] = model_alias
            yield f"data: {json.dumps(chunk)}\n\n"

        yield "data: [DONE]\n\n"

    async def __call__(self, request_dict: dict):
        return await self.generate(request_dict)


# ---------------------------------------------------------------------------
# Router (ingress)
# ---------------------------------------------------------------------------
@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": 1},
    max_ongoing_requests=200,
)
class MobileModelRouter:
    def __init__(self, **model_handles):
        self.model_handles = model_handles
        logger.info(
            f"Router ready with {len(model_handles)} models: "
            f"{sorted(model_handles.keys())}"
        )

    def _resolve_handle(self, model_name: str):
        key = model_name.replace("-", "_").replace(".", "_")
        return self.model_handles.get(key)

    async def __call__(self, request: Request):
        path = request.url.path.rstrip("/")

        if path.endswith("/models") and request.method == "GET":
            return self._list_models()

        if path.endswith("/health"):
            return JSONResponse({"status": "ok"})

        if path.endswith("/chat/completions") and request.method == "POST":
            try:
                body = await request.json()
            except Exception:
                return JSONResponse(
                    {"error": {"message": "Invalid JSON", "type": "invalid_request_error"}},
                    status_code=400,
                )

            model_name = body.get("model", "")
            handle = self._resolve_handle(model_name)

            if handle is None:
                return JSONResponse(
                    {
                        "error": {
                            "message": f"Model '{model_name}' not found. Available: {sorted(MODEL_CONFIGS.keys())}",
                            "type": "invalid_request_error",
                        }
                    },
                    status_code=404,
                )

            stream = body.get("stream", False)
            if stream:
                gen = handle.generate_stream.options(stream=True).remote(body)

                async def _sse():
                    async for chunk in gen:
                        yield chunk

                return StreamingResponse(_sse(), media_type="text/event-stream")
            else:
                result = await handle.remote(body)
                return JSONResponse(result)

        return JSONResponse(
            {"error": {"message": "Not found", "type": "invalid_request_error"}},
            status_code=404,
        )

    def _list_models(self):
        data = []
        for name, cfg in MODEL_CONFIGS.items():
            data.append({
                "id": name,
                "object": "model",
                "created": 0,
                "owned_by": "tii",
                "meta": {
                    "repo_id": cfg["repo_id"],
                    "filename": cfg["filename"],
                    "max_context_length": cfg["ctx"],
                    "default_max_tokens": cfg["max_tokens"],
                },
            })
        return JSONResponse({"object": "list", "data": data})


# ---------------------------------------------------------------------------
# Bind the deployment graph
# ---------------------------------------------------------------------------
_model_deployments = {}
for _name, _cfg in MODEL_CONFIGS.items():
    _safe = _name.replace("-", "_").replace(".", "_")
    _model_deployments[_safe] = FalconH1Model.options(
        name=_name,
        ray_actor_options={
            "num_cpus": 1,
            "num_gpus": _cfg["num_gpus"] if USE_GPU else 0,
        },
    ).bind(_name, _cfg)

app = MobileModelRouter.bind(**_model_deployments)
