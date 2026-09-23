from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

from comfyui_history import ComfyUIHistoryRunner, history_failed
from comfyui_http import ComfyUIHttpClient
from comfyui_sizes import resolve_output_size
from comfyui_workflows import MAX_EDIT_IMAGES, build_edit_workflow
from PIL import Image, ImageOps


def edit_payload(
    config: dict[str, Any],
    image_outputs: Path,
    resolve_images: Callable[[], list[Path]],
    args: Any,
    prompt: str,
) -> dict[str, Any]:
    images = [path for path in (resolve_images() or []) if path]
    if not images:
        return {"ok": False, "error": "no_input_image"}
    kept = images[:MAX_EDIT_IMAGES]
    truncated = len(images) - len(kept)
    client = ComfyUIHttpClient(config)
    uploaded: list[str] = []
    for image in kept:
        prepared = _prepare_upload_image(image, int(config.get("max_image_side", 1024)))
        uploaded.append(client.upload_image(prepared))
    steps = int(args.steps or config.get("steps", 25))
    cfg = float(args.cfg or config.get("cfg", 1.0))
    seed = int(args.seed if args.seed is not None else random.randint(1, 2**32 - 1))
    negative_prompt = str(args.negative_prompt or config.get("negative_prompt", ""))
    output_size = resolve_output_size(config, len(kept))
    try:
        prompt_body = build_edit_workflow(
            config, prompt, uploaded, steps, cfg, seed, negative_prompt, output_size
        )
    except ValueError as exc:
        return {
            "ok": False,
            "error": str(exc),
            "inputs": [str(path) for path in kept],
            "uploaded_images": uploaded,
            "truncated_inputs": truncated,
        }
    prompt_id, history = _run_prompt(config, image_outputs, prompt_body)
    status_payload = history_failed(history)
    if status_payload:
        return {
            "ok": False,
            "error": "workflow_failed",
            "prompt_id": prompt_id,
            "status": status_payload,
        }
    outputs, raw_image_count = _save_history_images(config, image_outputs, history)
    return {
        "ok": bool(outputs),
        "operation": "qwen_edit",
        "prompt_id": prompt_id,
        "inputs": [str(path) for path in kept],
        "uploaded_images": uploaded,
        "truncated_inputs": truncated,
        "seed": seed,
        "steps": steps,
        "cfg": cfg,
        "output_size": list(output_size) if output_size else None,
        "outputs": [str(path) for path in outputs],
        "raw_image_count": raw_image_count,
        "error": None if outputs else "no image found in history",
    }


def _run_prompt(
    config: dict[str, Any], image_outputs: Path, prompt_body: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    return ComfyUIHistoryRunner(config, image_outputs).run_prompt(prompt_body)


def _save_history_images(
    config: dict[str, Any], image_outputs: Path, history: dict[str, Any]
) -> tuple[list[Path], int]:
    return ComfyUIHistoryRunner(config, image_outputs).save_history_images(history)


def _prepare_upload_image(path: Path, max_side: int) -> Path:
    """Downscale oversized inputs client-side to protect VRAM and queue time.

    Args:
        path: Local source image path.
        max_side: Maximum allowed long edge in pixels. Values <= 0 disable
            downscaling.

    Returns:
        Path to upload: the original file, or a downscaled sibling copy.
    """
    if max_side <= 0:
        return path
    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            width, height = image.size
            if max(width, height) <= max_side:
                return path
            scale = max_side / max(width, height)
            resized = image.convert("RGB").resize(
                (max(64, int(width * scale)), max(64, int(height * scale))),
                Image.LANCZOS,
            )
    except Exception:
        return path
    target = path.with_name(f"{path.stem}_qwen{max_side}.png")
    try:
        resized.save(target)
    except Exception:
        return path
    return target
