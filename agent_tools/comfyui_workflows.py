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

#: Hard limit of the ``TextEncodeQwenImageEditPlus`` node (image1..image3).
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
) -> dict[str, Any]:
    """Build the Qwen-Image-2.1 reference-edit workflow graph.

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

    Returns:
        ComfyUI API-format prompt graph.

    Raises:
        ValueError: No image names were provided.
    """
    names = [str(name) for name in (image_names or []) if str(name).strip()]
    if not names:
        raise ValueError("qwen21_edit_workflow requires at least one image")
    names = names[:MAX_EDIT_IMAGES]

    load_nodes: dict[str, dict[str, Any]] = {
        "10": {"class_type": "LoadImage", "inputs": {"image": names[0]}},
    }
    edit_inputs: dict[str, Any] = {
        "clip": ["45", 0],
        "prompt": prompt,
        "vae": ["15", 0],
        "image1": ["10", 0],
    }
    # Extra reference images get their own LoadImage nodes ("20", "21").
    for index, extra in enumerate(names[1:], start=2):
        node_id = str(18 + index)  # 20, 21
        load_nodes[node_id] = {
            "class_type": "LoadImage",
            "inputs": {"image": extra},
        }
        edit_inputs[f"image{index}"] = [node_id, 0]

    graph: dict[str, Any] = {
        "44": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": config.get(
                    "unet_name", "qwen_image_2.1_nvfp4.safetensors"
                ),
                "weight_dtype": "default",
            },
        },
        "45": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": config.get(
                    "clip_name", "qwen3vl_8b_nvfp4_heretic.safetensors"
                ),
                "type": "qwen_image",
                "device": "default",
            },
        },
        "15": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": config.get(
                    "vae_name", "qwen_image_2.1_vae_bf16.safetensors"
                )
            },
        },
        "11": {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": edit_inputs,
        },
        "12": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative_prompt, "clip": ["45", 0]},
        },
        "30": {
            "class_type": "VAEEncode",
            "inputs": {"pixels": ["10", 0], "vae": ["15", 0]},
        },
        "19": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["44", 0],
                "positive": ["11", 0],
                "negative": ["12", 0],
                "latent_image": ["30", 0],
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": config.get("sampler_name", "euler"),
                "scheduler": config.get("scheduler", "simple"),
                # The edit model preserves the target via conditioning,
                # not via denoise. Always run full denoise here.
                "denoise": 1.0,
            },
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["19", 0], "vae": ["15", 0]},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"images": ["8", 0], "filename_prefix": "astrbot/qwen"},
        },
    }
    graph.update(load_nodes)
    return graph


def build_edit_workflow(
    config: dict[str, Any],
    prompt: str,
    image_names: list[str],
    steps: int,
    cfg: float,
    seed: int,
    negative_prompt: str = "",
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
        config, prompt, image_names, steps, cfg, seed, negative_prompt
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


def _node_ids_by_class(
    workflow_body: dict[str, Any], class_type: str
) -> list[str]:
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


def _set_text_input(
    workflow_body: dict[str, Any], node_id: str, value: str
) -> None:
    node = workflow_body.get(str(node_id))
    if not isinstance(node, dict):
        raise ValueError(f"custom_workflow_node_not_found: {node_id}")
    inputs = node.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError(f"custom_workflow_node_has_no_inputs: {node_id}")
    inputs["text"] = value
