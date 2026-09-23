"""Tier-A fixed edit templates for high-frequency requests.

These templates skip the LLM entirely: a keyword router picks the edit type
and the user's own words fill the CHANGE clause of the canonical
keep/preserve skeleton. Templates stay in code (not in user config) because
their wording is coupled to the Qwen-Image-2.1 edit node's behavior.
"""

from __future__ import annotations

import re

TEMPLATE_SKELETON = (
    "Keep the character and pose in <image1> unchanged, {change}, preserve "
    "the original facial features, hair, body shape and pose, {fit}, keep "
    "the original background and original lighting, clean anime "
    "illustration, cel-shaded, sharp details"
)

# The default outfit prompt from the validated two-image ComfyUI workflow.
TWO_IMAGE_OUTFIT_PROMPT = (
    "Put the clothing from <image2> onto the character in <image1>, replacing "
    "the outfit they are currently wearing. Keep the character's face, hairstyle, "
    "body shape and pose exactly as they appear in <image1>. Reproduce the "
    "clothing from <image2> faithfully: the same garment, the same colours, "
    "the same pattern, the same details. The clothing should fit the character's "
    "body naturally, with correct proportions, believable fabric drape and "
    "natural folds at the shoulders, elbows and waist. Keep the original "
    "background, camera angle, lighting and art style of <image1> unchanged."
)

_STRIP_CHARS = " ，,：:。.!！?？~～"

_CHANGE_PATTERNS = (
    re.compile(r"把(.+?)换成(.+)"),
    re.compile(r"[穿换]上(.+)"),
    re.compile(r"换(?:成)?(.+?)(?:衣服|服装|裙子|制服|外套|上衣|裤子|鞋子)$"),
    re.compile(
        r"(?:表情|背景|姿势|姿态|发色|头发|颜色|风格|衣服|服装)(?:换成|变成|改为|改成)(.+)"
    ),
)


def _clean_detail(text: str) -> str:
    return str(text or "").strip().strip(_STRIP_CHARS).strip()


def _extract_change(prompt: str) -> str:
    """Pull the CHANGE detail out of a user request.

    Args:
        prompt: Raw user requirement text.

    Returns:
        Extracted detail, or the whole cleaned prompt as fallback.
    """
    text = _clean_detail(prompt)
    for pattern in _CHANGE_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        groups = [part for part in match.groups() if _clean_detail(part)]
        if groups:
            return _clean_detail(groups[-1])
    return text


_MENTION_ONLY_LEFT = {"的衣服", "衣服", "的服装", "服装", "衣", "装", ""}


def _strip_mention_tokens(text: str) -> str:
    """Remove bare image-mention tokens, keeping real descriptions.

    Args:
        text: Extracted CHANGE detail.

    Returns:
        Detail without 图N / 第N张 / image N tokens.
    """
    result = re.sub(r"图\s*[一二三123]", "", str(text or ""))
    result = re.sub(r"第\s*[一二三123]\s*张", "", result)
    result = re.sub(r"image\s*[123]", "", result, flags=re.IGNORECASE)
    return _clean_detail(result)


def _outfit_change(prompt: str, image_count: int) -> str:
    detail = _extract_change(prompt)
    bare = detail == _clean_detail(prompt) and len(detail) <= 6
    mentions_ref = "图二" in prompt or "图2" in prompt or "image2" in prompt.lower()
    if image_count > 1 and (bare or mentions_ref):
        base = "put the outfit from <image2> on the character"
        extra = _strip_mention_tokens(detail)
        if extra and extra not in _MENTION_ONLY_LEFT:
            return f"{base} ({extra})"
        return base
    if bare:
        return f"change the outfit as requested: {detail or _clean_detail(prompt)}"
    return f"put {detail} on the character"


_TEMPLATES: tuple[dict[str, object], ...] = (
    {
        "name": "outfit",
        "keywords": (
            "换装",
            "换衣服",
            "穿上",
            "换上",
            "换件",
            "服装",
            "制服",
            "outfit",
        ),
        "change": _outfit_change,
        "fit": "reproduce the outfit from <image2> faithfully: the same garment, the same colours, the same pattern, the same details; do not keep any garment of the original outfit",
    },
    {
        "name": "expression",
        "keywords": (
            "换表情",
            "表情",
            "微笑",
            "大笑",
            "哭",
            "生气",
            "害羞",
            "惊讶",
            "expression",
        ),
        "change": lambda prompt, image_count: (
            f"change the expression to {_extract_change(prompt)}, "
            "keep everything else identical"
        ),
        "fit": "the new expression blends naturally with the face",
    },
    {
        "name": "background",
        "keywords": ("换背景", "背景", "换场景", "background"),
        "change": lambda prompt, image_count: (
            f"replace the background with {_extract_change(prompt)}, "
            "keep the character unchanged"
        ),
        "fit": "the character is lit consistently with the new background",
    },
    {
        "name": "pose",
        "keywords": (
            "换姿势",
            "换姿态",
            "姿势",
            "姿态",
            "坐下",
            "站起",
            "躺下",
            "pose",
        ),
        "change": lambda prompt, image_count: (
            f"change the pose to {_extract_change(prompt)}, "
            "keep the face and outfit unchanged"
        ),
        "fit": "anatomy stays natural and coherent",
    },
    {
        "name": "hair",
        "keywords": ("改发色", "发色", "染发", "换发色", "头发颜色", "hair"),
        "change": lambda prompt, image_count: (
            f"change the hair colour to {_extract_change(prompt)}, "
            "keep the hairstyle and everything else unchanged"
        ),
        "fit": "the new hair colour shades naturally with light and shadow",
    },
    {
        "name": "prop",
        "keywords": ("加", "拿着", "手持", "放一个", "抱着", "戴上", "prop"),
        "change": lambda prompt, image_count: (
            f"place {_extract_change(prompt)} naturally in the scene"
        ),
        "fit": "the object sits with correct scale and contact shadows",
    },
    {
        "name": "style",
        "keywords": ("风格化", "赛博", "水彩", "油画", "像素", "水墨", "style"),
        "change": lambda prompt, image_count: (
            f"restyle the whole image as {_extract_change(prompt)}"
        ),
        "fit": "the composition and subject stay recognizable",
    },
)


def match_template(user_prompt: str, image_count: int = 1) -> tuple[str, str] | None:
    """Match a user request to a fixed template.

    Args:
        user_prompt: Raw user requirement text.
        image_count: Number of supplied images (drives <image2> defaults).

    Returns:
        Tuple of template name and finished prompt, or None on no match.
    """
    text = str(user_prompt or "")
    for template in _TEMPLATES:
        keywords = template["keywords"]
        assert isinstance(keywords, tuple)
        if not any(keyword in text for keyword in keywords):
            continue
        change_builder = template["change"]
        assert callable(change_builder)
        count = max(1, int(image_count or 1))
        change = change_builder(text, count)
        fit = str(template["fit"])
        name = str(template["name"])
        if name == "outfit" and count > 1:
            detail = _strip_mention_tokens(_extract_change(text))
            if detail == _clean_detail(text) and len(detail) <= 6:
                detail = ""
            if detail and detail not in _MENTION_ONLY_LEFT:
                return (
                    name,
                    f"{TWO_IMAGE_OUTFIT_PROMPT} Additional outfit detail: {detail}.",
                )
            return name, TWO_IMAGE_OUTFIT_PROMPT
        if name == "outfit":
            fit = "the new outfit fits naturally with believable fabric folds"
        return name, TEMPLATE_SKELETON.format(change=change, fit=fit)
    return None


def template_names() -> list[str]:
    """List available fixed template names.

    Returns:
        Template names in match-priority order.
    """
    return [str(template["name"]) for template in _TEMPLATES]
