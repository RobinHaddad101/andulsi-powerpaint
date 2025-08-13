import base64
import io
import os
import random
from typing import Any, Dict

import numpy as np
import torch
from PIL import Image

import runpod
from transformers import CLIPTextModel

from powerpaint.models import BrushNetModel, UNet2DConditionModel
from powerpaint.pipelines import StableDiffusionPowerPaintBrushNetPipeline


# ------------------------------
# Model bootstrap (global)
# ------------------------------
HF_PPT2_REPO = os.environ.get("PPT2_REPO", "JunhaoZhuang/PowerPaint-v2-1")
BASE_MODEL_REPO = os.environ.get("BASE_MODEL", "runwayml/stable-diffusion-v1-5")
WEIGHT_DTYPE = torch.float16 if os.environ.get("WEIGHT_DTYPE", "float16") == "float16" else torch.float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Lazily initialized global pipeline
PIPE: StableDiffusionPowerPaintBrushNetPipeline | None = None


def _ensure_pipe() -> StableDiffusionPowerPaintBrushNetPipeline:
    global PIPE
    if PIPE is not None:
        return PIPE

    torch.set_grad_enabled(False)

    unet = UNet2DConditionModel.from_pretrained(
        BASE_MODEL_REPO, subfolder="unet", torch_dtype=WEIGHT_DTYPE
    ).to(DEVICE)

    brushnet = BrushNetModel.from_pretrained(
        HF_PPT2_REPO, subfolder="brushnet", torch_dtype=WEIGHT_DTYPE
    ).to(DEVICE)

    text_encoder = CLIPTextModel.from_pretrained(
        HF_PPT2_REPO, subfolder="text_encoder", torch_dtype=WEIGHT_DTYPE
    )

    pipe = StableDiffusionPowerPaintBrushNetPipeline.from_pretrained(
        BASE_MODEL_REPO,
        unet=unet,
        brushnet=brushnet,
        text_encoder=text_encoder,
        torch_dtype=WEIGHT_DTYPE,
        safety_checker=None,
    )

    # Task prompt tokens used by PowerPaint training
    pipe.add_tokens(
        placeholder_tokens=["P_obj", "P_ctxt", "P_shape"],
        initializer_tokens=["a", "a", "a"],
        num_vectors_per_token=[10, 10, 10],
        initialize_parameters=False,
    )

    pipe = pipe.to(DEVICE)
    PIPE = pipe
    return PIPE


# ------------------------------
# Utilities
# ------------------------------

def _b64_to_pil(data_b64: str) -> Image.Image:
    raw = base64.b64decode(data_b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _resize_divisible_by_8(img: Image.Image, max_side: int | None) -> Image.Image:
    w, h = img.size
    if max_side is not None:
        scale = min(max_side / max(w, h), 1.0)
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            w, h = img.size
    w, h = (w // 8) * 8, (h // 8) * 8
    if img.size != (w, h):
        img = img.resize((w, h), Image.LANCZOS)
    return img


# ------------------------------
# Prompt presets matching PPT2 in app.py
# ------------------------------

def _build_prompts(task: str, user_prompt: str, user_negative: str) -> Dict[str, str]:
    task = (task or "object-removal").strip()
    user_prompt = user_prompt or ""
    user_negative = user_negative or ""

    if task == "text-guided":
        return {
            "prompt": f"{user_prompt}",
            "negative_prompt": f"{user_negative}, worst quality, low quality, normal quality, bad quality, blurry",
            "promptA": "P_obj",
            "promptB": "P_obj",
            "negative_promptA": "P_obj",
            "negative_promptB": "P_obj",
        }
    elif task == "image-outpainting":
        return {
            "prompt": f"{user_prompt} empty scene blur",
            "negative_prompt": f"{user_negative}, worst quality, low quality, normal quality, bad quality, blurry",
            "promptA": "P_ctxt",
            "promptB": "P_ctxt",
            "negative_promptA": "P_obj",
            "negative_promptB": "P_obj",
        }
    elif task == "shape-guided":
        return {
            "prompt": f"{user_prompt}",
            "negative_prompt": f"{user_negative}, worst quality, low quality, normal quality, bad quality, blurry",
            "promptA": "P_shape",
            "promptB": "P_ctxt",
            "negative_promptA": "P_shape",
            "negative_promptB": "P_ctxt",
        }
    else:  # object-removal default
        return {
            "prompt": f"{user_prompt} empty scene blur".strip(),
            "negative_prompt": f"{user_negative}, worst quality, low quality, normal quality, bad quality, blurry".strip(
                ", "
            ),
            "promptA": "P_ctxt",
            "promptB": "P_ctxt",
            "negative_promptA": "P_obj",
            "negative_promptB": "P_obj",
        }


# ------------------------------
# Runpod handler
# ------------------------------

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    pipe = _ensure_pipe()

    inputs = event.get("input", {})
    image_b64: str = inputs.get("image")
    mask_b64: str = inputs.get("mask")
    task: str = inputs.get("task", "object-removal")
    prompt: str = inputs.get("prompt", "")
    negative_prompt: str = inputs.get("negative_prompt", "")
    steps: int = int(inputs.get("steps", 45))
    guidance_scale: float = float(inputs.get("guidance_scale", 10.0))
    tradeoff: float = float(inputs.get("fitting_degree", 1.0))
    max_side: int | None = inputs.get("max_side")
    if isinstance(max_side, str) and max_side.isdigit():
        max_side = int(max_side)
    elif isinstance(max_side, (int, float)):
        max_side = int(max_side)
    else:
        max_side = 768

    seed = inputs.get("seed")
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    seed = int(seed)

    if not image_b64 or not mask_b64:
        return {"error": "Both 'image' and 'mask' base64 strings are required."}

    # Decode inputs
    image = _b64_to_pil(image_b64)
    mask = _b64_to_pil(mask_b64)

    # Ensure same size
    if image.size != mask.size:
        mask = mask.resize(image.size, Image.NEAREST)

    # Resize for memory and ensure divisible by 8
    image_resized = _resize_divisible_by_8(image, max_side=max_side)
    mask_resized = mask.resize(image_resized.size, Image.NEAREST)

    # Build masked image (black holes)
    hole_value = (0, 0, 0)
    masked_image = Image.composite(Image.new("RGB", image_resized.size, hole_value), image_resized, mask_resized.convert("L"))

    # Set prompts
    prompts = _build_prompts(task, prompt, negative_prompt)

    # Inference
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    with torch.autocast(DEVICE if DEVICE != "cpu" else "cpu"):
        result = pipe(
            promptA=prompts["promptA"],
            promptB=prompts["promptB"],
            prompt=prompts["prompt"],
            negative_promptA=prompts["negative_promptA"],
            negative_promptB=prompts["negative_promptB"],
            negative_prompt=prompts["negative_prompt"],
            tradeoff=tradeoff,
            image=masked_image,
            mask=mask_resized,
            num_inference_steps=steps,
            generator=g,
            brushnet_conditioning_scale=1.0,
            guidance_scale=guidance_scale,
            width=image_resized.size[0],
            height=image_resized.size[1],
        ).images[0]

    # Paste the inpainting result back to resized original using (optionally augmented) mask
    result_paste = Image.composite(result, image_resized, mask_resized.convert("L"))

    # If we resized, optionally upscale back to original size for output
    if result_paste.size != image.size:
        result_paste = result_paste.resize(image.size, Image.LANCZOS)

    return {
        "output": _pil_to_b64(result_paste),
        "seed": seed,
        "width": result_paste.size[0],
        "height": result_paste.size[1],
    }


runpod.serverless.start({"handler": handler}) 