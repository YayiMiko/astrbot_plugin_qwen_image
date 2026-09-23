import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from comfyui_command_runner import run_cli_action
from comfyui_inputs import ComfyUIImageResolver
from comfyui_operations import edit_payload
from comfyui_sizes import allowed_sizes
from comfyui_status import build_status_payload
from comfyui_workflows import describe_custom_workflow
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _find_astrbot_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (
            (parent / "main.py").exists()
            and (parent / "data").is_dir()
            and (parent / "astrbot").is_dir()
        ):
            return parent
    return Path(__file__).resolve().parents[1]


ROOT = _find_astrbot_root()
WORKSPACE = ROOT / "workspace"
OUTPUTS = WORKSPACE / "outputs"
IMAGE_OUTPUTS = OUTPUTS / "images"
CONFIG = ROOT / "data" / "config" / "astrbot_plugin_qwen_image_config.json"
IMAGE_RESOLVER = ComfyUIImageResolver(WORKSPACE)


DEFAULT_CONFIG = {
    "comfyui_base_url": "http://127.0.0.1:8188",
    "workflow": "qwen21_edit",
    "output_size_mode": "target",
    "output_aspect": "4:3",
    "output_megapixels": 1.0,
    "single_image_size_mode": "target",
    "limit_image_megapixels": True,
    "timeout": 600,
    "poll_interval": 2,
    "allowed_sizes": [
        "832x1216",
        "896x1152",
        "1024x1024",
        "1152x896",
        "1216x832",
    ],
    "steps": 25,
    "cfg": 1.0,
    "sampler_name": "euler",
    "scheduler": "simple",
    "unet_name": "qwen_image_2.1_nvfp4.safetensors",
    "clip_name": "qwen3vl_8b_nvfp4_heretic.safetensors",
    "vae_name": "qwen_image_2.1_vae_bf16.safetensors",
    "negative_prompt": "",
    "max_image_side": 0,
    "custom_workflow_enabled": False,
    "custom_workflow_path": "",
    "custom_workflow_json": "",
}


def _json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def load_config() -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    config.update(_flatten_config(_json_file(CONFIG)))
    return config


def _flatten_config(config: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in dict(config or {}).items():
        if str(key).startswith("qwen_") and isinstance(value, dict):
            flat.update(value)
    for key, value in dict(config or {}).items():
        if not (str(key).startswith("qwen_") and isinstance(value, dict)):
            flat[key] = value
    return flat


def _recent_images(limit: int) -> list[Path]:
    return IMAGE_RESOLVER.recent_images(limit)


def _latest_image() -> Path:
    return IMAGE_RESOLVER.latest_image()


def resolve_image(value: str | None) -> Path:
    return IMAGE_RESOLVER.resolve_image(value)


def resolve_images(values: list[str] | None) -> list[Path]:
    seen: list[Path] = []
    for value in values or []:
        try:
            path = resolve_image(value)
        except Exception:
            continue
        if path not in seen:
            seen.append(path)
    return seen


def result(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def status(args) -> None:
    config = load_config()
    size_options = [
        f"{width}x{height}"
        for width, height in allowed_sizes(config, DEFAULT_CONFIG["allowed_sizes"])
    ]
    payload = build_status_payload(config, size_options)
    if bool(config.get("custom_workflow_enabled", False)):
        payload["workflow_mode"] = "custom"
        payload["custom_binding"] = describe_custom_workflow(config)
    else:
        payload["workflow_mode"] = "builtin"
    result(payload)


def recent(args) -> None:
    images = _recent_images(args.limit)
    payload = {"ok": True, "count": len(images), "images": []}
    for path in images:
        item = {
            "path": str(path),
            "relative_path": str(path.relative_to(WORKSPACE)),
            "size": path.stat().st_size,
            "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(
                timespec="seconds"
            ),
        }
        try:
            with Image.open(path) as img:
                item["width"] = img.width
                item["height"] = img.height
                item["format"] = img.format
        except Exception:
            pass
        payload["images"].append(item)
    result(payload)


def edit(args) -> None:
    config = load_config()
    prompt = str(args.prompt or "").strip()
    if not prompt:
        result({"ok": False, "error": "missing_prompt", "message": "缺少提示词"})
        return

    inputs = list(args.input or []) or ["latest"]
    result(
        run_cli_action(
            lambda: edit_payload(
                config, IMAGE_OUTPUTS, lambda: resolve_images(inputs), args, prompt
            )
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="AstrBot Qwen-Image 助手")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("status")
    p.set_defaults(func=status)

    p = sub.add_parser("recent")
    p.add_argument("--limit", type=int, default=5)
    p.set_defaults(func=recent)

    p = sub.add_parser("edit")
    p.add_argument("--prompt", required=True)
    p.add_argument("--input", action="append", default=None)
    p.add_argument("--steps", type=int)
    p.add_argument("--cfg", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--negative-prompt")
    p.set_defaults(func=edit)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
