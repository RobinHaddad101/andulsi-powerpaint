import base64
import io
import os
import random
from typing import Any, Dict
from safetensors.torch import load_file
import numpy as np
import torch
from PIL import Image, ImageFilter # Import ImageFilter

import runpod
from transformers import CLIPTextModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_model

from powerpaint.models import BrushNetModel, UNet2DConditionModel
from powerpaint.pipelines import StableDiffusionPowerPaintBrushNetPipeline
from powerpaint.utils.utils import TokenizerWrapper, add_tokens


# ------------------------------
# Model bootstrap (global)
# ------------------------------
HF_PPT2_REPO = os.environ.get("PPT2_REPO", "JunhaoZhuang/PowerPaint-v2-1")
# Use the base pipeline shipped in the repo
HF_PPT2_BASE_SUBFOLDER = os.environ.get("PPT2_BASE_SUBFOLDER", "realisticVisionV60B1_v51VAE")
WEIGHT_DTYPE = torch.float16 if os.environ.get("WEIGHT_DTYPE", "float16") == "float16" else torch.float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Lazily initialized global pipeline
PIPE: StableDiffusionPowerPaintBrushNetPipeline | None = None


def _ensure_pipe() -> StableDiffusionPowerPaintBrushNetPipeline:
    global PIPE
    if PIPE is not None:
        return PIPE

    torch.set_grad_enabled(False)
    # 1) Download just the base pipeline subfolder
    from huggingface_hub import snapshot_download
    snapshot_dir = snapshot_download(
        HF_PPT2_REPO,
        allow_patterns=[f"{HF_PPT2_BASE_SUBFOLDER}/*"],
    )
    base_path = os.path.join(snapshot_dir, HF_PPT2_BASE_SUBFOLDER)



    # Base components from the repo's pipeline
    unet = UNet2DConditionModel.from_pretrained(base_path, subfolder="unet", torch_dtype=WEIGHT_DTYPE)

    # Build BrushNet and load weights
    brushnet = BrushNetModel.from_unet(unet)
    brushnet_weights = hf_hub_download(
        repo_id=HF_PPT2_REPO,
        filename="diffusion_pytorch_model.safetensors",
        subfolder="PowerPaint_Brushnet",
    )

    bn_state = load_file(brushnet_weights)
    bn_target = brushnet.state_dict()
    bn_state = {k: v for k, v in bn_state.items() if k in bn_target and v.shape == bn_target[k].shape}
    brushnet.load_state_dict(bn_state, strict=False)


    # BrushNet's dedicated text encoder
    text_encoder_brushnet = CLIPTextModel.from_pretrained(base_path, subfolder="text_encoder", torch_dtype=WEIGHT_DTYPE)
    te_brushnet_weights = hf_hub_download(
        repo_id=HF_PPT2_REPO,
        filename="pytorch_model.bin",
        subfolder="PowerPaint_Brushnet",
    )
    text_encoder_brushnet.load_state_dict(torch.load(te_brushnet_weights, map_location="cpu"), strict=False)

    # Build the pipeline from the repo's base folder (model_index.json present)
    pipe = StableDiffusionPowerPaintBrushNetPipeline.from_pretrained(
        base_path,
        unet=unet,
        brushnet=brushnet,
        text_encoder_brushnet=text_encoder_brushnet,
        torch_dtype=WEIGHT_DTYPE,
        safety_checker=None,
    )

    # Add learned task tokens into tokenizer and BrushNet text encoder
    pipe.tokenizer = TokenizerWrapper(from_pretrained=base_path, subfolder="tokenizer")

    add_tokens(
        tokenizer=pipe.tokenizer,
        text_encoder=pipe.text_encoder_brushnet,
        placeholder_tokens=["P_obj", "P_ctxt", "P_shape"],
        initialize_tokens=["a", "a", "a"],
        num_vectors_per_token=10,
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
        # Note: app.py adds "empty scene" for ppt-v2 which maps to BrushNet pipeline
        # Here, we only have BrushNet pipeline, so we directly add "empty scene"
        return {
            "prompt": f"{user_prompt} empty scene",
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
    task: str = inputs.get("task", "object-removal")
    prompt: str = inputs.get("prompt", "")
    negative_prompt: str = inputs.get("negative_prompt", "")
    steps: int = int(inputs.get("steps", 45))
    guidance_scale: float = float(inputs.get("guidance_scale", 10.0))
    tradeoff: float = float(inputs.get("fitting_degree", 1.0))
    max_side: int | None = inputs.get("max_side")

    # Parse vertical_expansion_ratio and horizontal_expansion_ratio
    vertical_expansion_ratio: float = float(inputs.get("vertical_expansion_ratio", 1.0))
    horizontal_expansion_ratio: float = float(inputs.get("horizontal_expansion_ratio", 1.0))

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

    if not image_b64:
        return {"error": "'image' base64 string is required."}

    # Decode input image
    image = _b64_to_pil(image_b64)

    original_width, original_height = image.size

    # Handle image outpainting: create expanded image and mask
    if task == "image-outpainting":
        o_W, o_H = original_width, original_height
        c_W = int(horizontal_expansion_ratio * o_W)
        c_H = int(vertical_expansion_ratio * o_H)

        # Ensure output size is at least original image size if ratios are <= 1
        if c_W < o_W: c_W = o_W
        if c_H < o_H: c_H = o_H

        expand_img = np.ones((c_H, c_W, 3), dtype=np.uint8) * 127 # Grey background
        original_img_np = np.array(image)
        
        # Calculate paste coordinates to center the original image
        paste_x = int((c_W - o_W) / 2.0)
        paste_y = int((c_H - o_H) / 2.0)

        expand_img[paste_y : paste_y + o_H, paste_x : paste_x + o_W, :] = original_img_np

        blurry_gap = 10 # from app.py
        expand_mask_np = np.ones((c_H, c_W, 3), dtype=np.uint8) * 255 # White mask (255 = masked)

        # Create a "hole" in the mask corresponding to the original image area,
        # but with a blurry gap at the edges, similar to app.py
        if vertical_expansion_ratio == 1 and horizontal_expansion_ratio != 1:
            expand_mask_np[paste_y : paste_y + o_H, paste_x + blurry_gap : paste_x + o_W - blurry_gap, :] = 0
        elif vertical_expansion_ratio != 1 and horizontal_expansion_ratio == 1:
            expand_mask_np[paste_y + blurry_gap : paste_y + o_H - blurry_gap, paste_x : paste_x + o_W, :] = 0
        else: # Covers horizontal and vertical expansion, and no expansion (if ratios are 1)
            expand_mask_np[paste_y + blurry_gap : paste_y + o_H - blurry_gap, paste_x + blurry_gap : paste_x + o_W - blurry_gap, :] = 0
        
        image_for_pipeline = Image.fromarray(expand_img)
        mask_for_pipeline = Image.fromarray(expand_mask_np)

        # Resize the expanded image and mask
        image_resized = _resize_divisible_by_8(image_for_pipeline, max_side=max_side)
        mask_resized = mask_for_pipeline.resize(image_resized.size, Image.NEAREST)

    else:
        # Original inpainting/object removal logic for other tasks
        mask_b64: str = inputs.get("mask")
        if not mask_b64:
            return {"error": "'mask' base64 string is required for tasks other than image-outpainting."}
        mask = _b64_to_pil(mask_b64)
        if image.size != mask.size:
            mask = mask.resize(image.size, Image.NEAREST)

        image_resized = _resize_divisible_by_8(image, max_side=max_side)
        mask_resized = mask.resize(image_resized.size, Image.NEAREST)

    # Build masked image (black holes) based on the *resized* mask and image
    # This is for the `image` argument of the pipeline which expects masked regions to be black.
    hole_value = (0, 0, 0)
    masked_image_for_pipeline = Image.composite(Image.new("RGB", image_resized.size, hole_value), image_resized, mask_resized.convert("L"))

    # Set prompts
    prompts = _build_prompts(task, prompt, negative_prompt)
    print(f"DEBUG: Prompts used: {prompts}") # Logging prompts for debugging

    # Inference
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    with torch.autocast(DEVICE if DEVICE != "cpu" else "cpu"):
        result = pipe(
            promptA=prompts["promptA"],
            promptB=prompts["promptB"],
            promptU=prompts["prompt"],
            negative_promptA=prompts["negative_promptA"],
            negative_promptB=prompts["negative_promptB"],
            negative_promptU=prompts["negative_prompt"],
            tradoff=tradeoff,
            tradoff_nag=tradeoff,
            image=masked_image_for_pipeline, # Use the prepared masked image
            mask=mask_resized, # Use the resized mask
            num_inference_steps=steps,
            generator=g,
            brushnet_conditioning_scale=1.0,
            guidance_scale=guidance_scale,
            width=image_resized.size[0],
            height=image_resized.size[1],
        ).images[0]
    
    # If we resized the initial image, upscale the result back to the target expanded size
    # For tasks other than outpainting, it should resize to original image size
    final_output_image_size = (c_W, c_H) if task == "image-outpainting" else (original_width, original_height)
    if result.size != final_output_image_size:
        result = result.resize(final_output_image_size, Image.LANCZOS)
    
    print(f"Outpainting completed. Result image size: {result.size}") # Log the final image size

    return {
        "output": _pil_to_b64(result), # Return the final outpainted/inpainted image
        "seed": seed,
        "width": result.size[0],
        "height": result.size[1],
    }


runpod.serverless.start({"handler": handler}) 