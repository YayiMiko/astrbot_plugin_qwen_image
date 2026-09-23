"""Qwen-Image-2.1 ComfyUI workflow builders (API format).

Two sources can produce the edit graph:

- Built-in ``qwen21_edit`` graph (validated against the local ComfyUI).
- User-supplied ComfyUI API-format JSON (custom workflow mode).

Custom graphs must honor the binding contract documented in `custom contract`
below. Future text-to-image support plugs into `build_edit_workflow` as a new
workflow name (e.g. ``qwen21_t2i``) without changing callers.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

#: Chat-side limit; ``TextEncodeQwenImage21`` supports these image slots.
MAX_EDIT_IMAGES = 3

#: Workflow names understood by `build_edit_workflow`. Unknown names raise
#: ``unsupported_workflow`` so future graphs (e.g. text-to-image) fail loudly
#: instead of silently running the wrong topology.
BUILTIN_EDIT_WORKFLOWS = ("qwen21_edit",)


def qwen21_edit_workflow(
    config: dict[str, Any],
    prompt: str,
    image_names: list[str],
    steps: int,
    cfg: float,
    seed: int,
    negative_prompt: str = "",
    output_size: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Build the API graph of the validated two-image reference workflow.

    Args:
        config: Plugin configuration with model filenames.
        prompt: Finished natural-language edit paragraph (may reference
            ``<image1>`` .. ``<image3>``).
        image_names: Uploaded ComfyUI image names, 1-3 entries.
            The first entry is always the edit target.
        steps: Sampling steps (25 quick / 40 final).
        cfg: CFG scale (1.0 default; raise only for legible text).
        seed: Random seed.
        negative_prompt: Negative prompt, normally empty. Only meaningful
            when cfg is raised for text rendering.
        output_size: Optional forced (width, height) for the independent output
            latent. Without it, use the encoder's image-sized latent.

    Returns:
        ComfyUI API-format prompt graph.

    Raises:
        ValueError: No image names were provided.
    """
    names = [str(name) for name in (image_names or []) if str(name).strip()]
    if not names:
        raise ValueError("qwen21_edit_workflow requires at least one image")
    names = names[:MAX_EDIT_IMAGES]

    graph: dict[str, Any] = {
        "451": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": config.get(
                    "unet_name", "qwen_image_2.1_nvfp4.safetensors"
                ),
                "weight_dtype": "default",
            },
        },
        "453": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": config.get(
                    "clip_name", "qwen3vl_8b_nvfp4_heretic.safetensors"
                ),
                "type": "qwen_image",
                "device": "default",
            },
        },
        "454": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": config.get(
                    "vae_name", "qwen_image_2.1_vae_bf16.safetensors"
                )
            },
        },
        "469": {
            "class_type": "QwenImage21Cache",
            "inputs": {"model": ["451", 0], "device": "auto", "dtype": "int8"},
        },
        "480": {"class_type": "PrimitiveFloat", "inputs": {"value": 1.0}},
    }
    edit_inputs: dict[str, Any] = {
        "clip": ["453", 0],
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "resolution": 0,
        "vae": ["454", 0],
    }
    # Cap references at 1 MiP by default; the first image also determines the
    # output latent when no explicit output size is configured.
    limit_image_megapixels = bool(config.get("limit_image_megapixels", True))
    for index, name in enumerate(names, start=1):
        load_id = ("470", "475", "490")[index - 1]
        size_id = ("483", "484", "491")[index - 1]
        math_id = ("485", "486", "492")[index - 1]
        scale_id = ("477", "479", "493")[index - 1]
        graph[load_id] = {"class_type": "LoadImage", "inputs": {"image": name}}
        if not limit_image_megapixels:
            edit_inputs[f"images.image_{index}"] = [load_id, 0]
            continue
        graph[size_id] = {
            "class_type": "GetImageSize",
            "inputs": {"image": [load_id, 0]},
        }
        graph[math_id] = {
            "class_type": "ComfyMathExpression",
            "inputs": {
                "expression": "min(c, a*b/1048576)",
                "values.a": [size_id, 0],
                "values.b": [size_id, 1],
                "values.c": ["480", 0],
            },
        }
        graph[scale_id] = {
            "class_type": "ImageScaleToTotalPixels",
            "inputs": {
                "upscale_method": "lanczos",
                "megapixels": [math_id, 0],
                "resolution_steps": 32,
                "image": [load_id, 0],
            },
        }
        edit_inputs[f"images.image_{index}"] = [scale_id, 0]

    graph["474"] = {"class_type": "TextEncodeQwenImage21", "inputs": edit_inputs}
    graph["458"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": ["469", 0],
            "positive": ["474", 0],
            "negative": ["474", 1],
            "latent_image": ["456", 0] if output_size else ["474", 2],
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": config.get("sampler_name", "euler"),
            "scheduler": config.get("scheduler", "simple"),
            "denoise": 1.0,
        },
    }
    graph["457"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["458", 0], "vae": ["454", 0]},
    }
    graph["461"] = {
        "class_type": "SaveImageAdvanced",
        "inputs": {
            "images": ["457", 0],
            "filename_prefix": "astrbot/qwen",
            "format": "png",
            "format.bit_depth": "8-bit",
            "format.input_color_space": "sRGB",
        },
    }
    if output_size:
        width, height = output_size
        graph["456"] = {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": int(width), "height": int(height), "batch_size": 1},
        }
    return graph


def build_edit_workflow(
    config: dict[str, Any],
    prompt: str,
    image_names: list[str],
    steps: int,
    cfg: float,
    seed: int,
    negative_prompt: str = "",
    output_size: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Build the effective edit graph (built-in or custom).

    Args:
        config: Plugin configuration.
        prompt: Finished edit paragraph.
        image_names: Uploaded ComfyUI image names (first = edit target).
        steps: Sampling steps.
        cfg: CFG scale.
        seed: Random seed.
        negative_prompt: Negative prompt (custom graphs only use it when a
            negative text node exists).
        output_size: Forced output size for the built-in graph. Custom
            graphs manage their own topology and ignore this.

    Returns:
        ComfyUI API-format prompt graph.

    Raises:
        ValueError: Unknown workflow name or invalid custom graph.
    """
    if bool(config.get("custom_workflow_enabled", False)):
        return custom_qwen_edit_workflow(
            config, prompt, image_names, steps, cfg, seed, negative_prompt
        )
    workflow_name = str(config.get("workflow") or "qwen21_edit")
    if workflow_name not in BUILTIN_EDIT_WORKFLOWS:
        raise ValueError(f"unsupported_workflow: {workflow_name}")
    return qwen21_edit_workflow(
        config, prompt, image_names, steps, cfg, seed, negative_prompt, output_size
    )


def custom_workflow_source(config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Load the user-supplied custom workflow dict.

    The file path (when set) takes precedence over the inline JSON text.

    Args:
        config: Plugin configuration.

    Returns:
        Tuple of source label and parsed graph dict.

    Raises:
        ValueError: No source configured, file missing, or invalid JSON.
    """
    path_text = str(config.get("custom_workflow_path") or "").strip()
    if path_text:
        path = Path(path_text).expanduser()
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[1] / path
        if not path.is_file():
            raise ValueError(f"custom_workflow_not_found: {path}")
        try:
            raw = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            raise ValueError(f"custom_workflow_invalid_json: {exc}") from exc
        return f"path:{path}", _unwrap_workflow_body(raw, f"path:{path}")
    inline_text = str(config.get("custom_workflow_json") or "").strip()
    if not inline_text:
        raise ValueError("custom_workflow_not_configured")
    try:
        raw = json.loads(inline_text)
    except Exception as exc:
        raise ValueError(f"custom_workflow_invalid_json: {exc}") from exc
    return "inline", _unwrap_workflow_body(raw, "inline")


def _unwrap_workflow_body(raw: Any, source: str) -> dict[str, Any]:
    body = (
        raw.get("prompt")
        if isinstance(raw, dict) and isinstance(raw.get("prompt"), dict)
        else raw
    )
    if not isinstance(body, dict) or not body:
        raise ValueError(f"custom_workflow_invalid_graph: {source}")
    return body


def _node_ids_by_class(workflow_body: dict[str, Any], class_type: str) -> list[str]:
    return sorted(
        str(node_id)
        for node_id, node in workflow_body.items()
        if isinstance(node, dict) and str(node.get("class_type") or "") == class_type
    )


def _text_encode_node_ids(workflow_body: dict[str, Any]) -> list[str]:
    node_ids: list[str] = []
    for node_id, node in workflow_body.items():
        if not isinstance(node, dict):
            continue
        class_type = str(node.get("class_type") or "")
        inputs = node.get("inputs")
        if (
            "TextEncode" in class_type
            and isinstance(inputs, dict)
            and isinstance(inputs.get("text"), str)
        ):
            node_ids.append(str(node_id))
    return sorted(node_ids)


def describe_custom_workflow(config: dict[str, Any]) -> dict[str, Any]:
    """Summarize a custom workflow for status output (never raises).

    Args:
        config: Plugin configuration.

    Returns:
        Dict with `ok`, binding counts, or an `error` code.
    """
    try:
        source, body = custom_workflow_source(config)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    load_ids = _node_ids_by_class(body, "LoadImage")
    text_ids = _text_encode_node_ids(body)
    save_ids = _node_ids_by_class(body, "SaveImage")
    if not save_ids:
        return {"ok": False, "error": "custom_workflow_no_save_node"}
    if not load_ids:
        return {"ok": False, "error": "custom_workflow_no_load_node"}
    if not text_ids:
        return {"ok": False, "error": "custom_workflow_positive_node_not_found"}
    return {
        "ok": True,
        "source": source,
        "load_slots": len(load_ids),
        "text_nodes": len(text_ids),
        "save_nodes": len(save_ids),
        "positive_node": text_ids[0],
        "negative_node": text_ids[1] if len(text_ids) > 1 else "",
    }


def custom_qwen_edit_workflow(
    config: dict[str, Any],
    prompt: str,
    image_names: list[str],
    steps: int,
    cfg: float,
    seed: int,
    negative_prompt: str = "",
) -> dict[str, Any]:
    """Bind runtime values into a user-supplied edit graph.

    Binding contract (documented for public users):

    - ``LoadImage`` nodes, in sorted node-id order, receive the uploaded
      images: first node = edit target, rest = references. The graph must
      contain at least as many ``LoadImage`` nodes as supplied images.
    - The first ``*TextEncode*`` node with a ``text`` input is the positive
      prompt; a second one (when present) is the negative prompt.
    - All ``SaveImage`` filename prefixes are rewritten to
      ``astrbot/qwen_custom`` so plugin outputs never mix with ComfyUI UI runs.
    - Every ``seed`` / ``noise_seed`` input is randomized per request.
    - ``VAEEncode`` target wiring is the graph author's responsibility; when
      it does not reference the first ``LoadImage`` node a warning is logged.

    Args:
        config: Plugin configuration.
        prompt: Finished edit paragraph.
        image_names: Uploaded ComfyUI image names (first = edit target).
        steps: Sampling steps (injected when the sampler exposes them).
        cfg: CFG scale (injected when the sampler exposes it).
        seed: Random seed.
        negative_prompt: Negative prompt for the second text node, if any.

    Returns:
        Bound ComfyUI API-format prompt graph.

    Raises:
        ValueError: Binding contract violated.
    """
    names = [str(name) for name in (image_names or []) if str(name).strip()]
    if not names:
        raise ValueError("custom_workflow_requires_images")
    _, body = custom_workflow_source(config)
    graph = copy.deepcopy(body)

    load_ids = _node_ids_by_class(graph, "LoadImage")
    if len(load_ids) < len(names):
        raise ValueError(
            "custom_workflow_image_binding_mismatch: "
            f"graph has {len(load_ids)} LoadImage nodes "
            f"for {len(names)} images"
        )
    for node_id, image_name in zip(load_ids, names):
        node = graph[node_id]
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            raise ValueError(f"custom_workflow_node_has_no_inputs: {node_id}")
        inputs["image"] = image_name

    text_ids = _text_encode_node_ids(graph)
    if not text_ids:
        raise ValueError("custom_workflow_positive_node_not_found")
    _set_text_input(graph, text_ids[0], prompt)
    if len(text_ids) > 1:
        _set_text_input(graph, text_ids[1], negative_prompt)

    for node_id, node in graph.items():
        if not isinstance(node, dict):
            continue
        class_type = str(node.get("class_type") or "")
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        if class_type == "SaveImage":
            if "filename_prefix" in inputs:
                inputs["filename_prefix"] = "astrbot/qwen_custom"
        if "seed" in inputs:
            inputs["seed"] = seed
        if "noise_seed" in inputs:
            inputs["noise_seed"] = seed
        if class_type == "KSampler":
            if "steps" in inputs:
                inputs["steps"] = steps
            if "cfg" in inputs:
                inputs["cfg"] = cfg

    if not _node_ids_by_class(graph, "SaveImage"):
        raise ValueError("custom_workflow_no_save_node")
    return graph


def _set_text_input(workflow_body: dict[str, Any], node_id: str, value: str) -> None:
    node = workflow_body.get(str(node_id))
    if not isinstance(node, dict):
        raise ValueError(f"custom_workflow_node_not_found: {node_id}")
    inputs = node.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError(f"custom_workflow_node_has_no_inputs: {node_id}")
    inputs["text"] = value
