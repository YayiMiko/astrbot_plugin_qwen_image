from __future__ import annotations

from typing import Any


def parse_size(value: Any) -> tuple[int, int] | None:
    if isinstance(value, str):
        text = value.strip().lower()
        for separator in ("x", "×", "*", "＊", "✕", "✖", "х"):
            text = text.replace(separator, "x")
        if "x" not in text:
            return None
        left, right = text.split("x", 1)
        try:
            width = int(left.strip())
            height = int(right.strip())
        except ValueError:
            return None
        if width > 0 and height > 0:
            return width, height
    if isinstance(value, dict):
        try:
            width = int(value.get("width"))
            height = int(value.get("height"))
        except (TypeError, ValueError):
            return None
        if width > 0 and height > 0:
            return width, height
    return None


def allowed_sizes(
    config: dict[str, Any], default_sizes: list[str]
) -> list[tuple[int, int]]:
    raw_sizes = config.get("allowed_sizes") or default_sizes
    sizes: list[tuple[int, int]] = []
    for item in raw_sizes if isinstance(raw_sizes, list) else []:
        parsed = parse_size(item)
        if parsed and parsed not in sizes:
            sizes.append(parsed)
    if not sizes:
        sizes = [item for item in (parse_size(item) for item in default_sizes) if item]
    return sizes


def generation_size(
    config: dict[str, Any], default_sizes: list[str], width: int, height: int
) -> tuple[int, int]:
    width = int(width)
    height = int(height)
    allowed = allowed_sizes(config, default_sizes)
    if allowed and (width, height) not in allowed:
        requested_ratio = width / height
        same_orientation = [
            size for size in allowed if (size[0] >= size[1]) == (width >= height)
        ] or allowed
        width, height = min(
            same_orientation,
            key=lambda size: (
                abs((size[0] / size[1]) - requested_ratio),
                abs(size[0] * size[1] - width * height),
            ),
        )
    return width, height


OUTPUT_ASPECTS = {
    "1:1": (1, 1),
    "3:4": (3, 4),
    "4:3": (4, 3),
    "2:3": (2, 3),
    "3:2": (3, 2),
    "9:16": (9, 16),
    "16:9": (16, 9),
}

#: Latent grid alignment, mirroring the UI node's 倍数 control.
SIZE_MULTIPLE = 32


def resolve_output_size(
    config: dict[str, Any],
    image_count: int = 2,
) -> tuple[int, int] | None:
    """Resolve a forced output size from aspect + megapixel settings.

    Args:
        config: Plugin configuration with `output_size_mode`,
            `single_image_size_mode`, `output_aspect`, and `output_megapixels`.
        image_count: Number of reference images supplied to the edit workflow.

    Returns:
        (width, height) snapped down to multiples of 32, or None when the
        output should follow the target image (`target` mode or bad config).
    """
    mode = (
        str(config.get("single_image_size_mode") or "target").strip().lower()
        if image_count == 1
        else str(config.get("output_size_mode") or "target").strip().lower()
    )
    if image_count == 1 and mode == "configured":
        mode = "aspect"
    if mode != "aspect":
        return None
    aspect = str(config.get("output_aspect") or "").strip()
    ratio = OUTPUT_ASPECTS.get(aspect)
    if ratio is None:
        return None
    try:
        megapixels = float(config.get("output_megapixels", 1.0))
    except (TypeError, ValueError):
        return None
    if megapixels <= 0:
        return None
    ar_w, ar_h = ratio
    width = int((megapixels * 1_000_000 * ar_w / ar_h) ** 0.5)
    height = int((megapixels * 1_000_000 * ar_h / ar_w) ** 0.5)
    width = max(SIZE_MULTIPLE, (width // SIZE_MULTIPLE) * SIZE_MULTIPLE)
    height = max(SIZE_MULTIPLE, (height // SIZE_MULTIPLE) * SIZE_MULTIPLE)
    return width, height
