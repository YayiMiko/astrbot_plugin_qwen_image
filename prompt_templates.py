"""Tier-A fixed edit templates for high-frequency requests.

These templates skip the LLM only for short, unambiguous Chinese edits.
Compound or multi-reference requests go to the vision-grounded rewriter.
"""

from __future__ import annotations

import re

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
    re.compile(r"(?:换成|改成)(.+?)表情$"),
    re.compile(r"(?:画成|改成|变成)(.+?)风格$"),
    re.compile(r"(?:让(?:她|他|人物|角色))?(坐下|站起|躺下)"),
    re.compile(r"(?:给.+?)?加(一[^，,。；;]+|个[^，,。；;]+)"),
    re.compile(r"把(.+?)换成(.+)"),
    re.compile(r"(?:画成|改成)([^，,。；;]+)$"),
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


_TEMPLATES: tuple[dict[str, object], ...] = (
    {
        "name": "outfit",
        "keywords": (
            "换装",
            "换衣服",
            "穿上",
            "换上",
            "换件",
            "衣服换成",
            "服装换成",
        ),
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
        ),
    },
    {
        "name": "background",
        "keywords": (
            "换背景",
            "背景换成",
            "背景改成",
            "换场景",
        ),
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
        ),
    },
    {
        "name": "hair",
        "keywords": (
            "改发色",
            "发色改成",
            "染发",
            "换发色",
            "头发颜色改成",
        ),
    },
    {
        "name": "prop",
        "keywords": (
            "加上",
            "加一",
            "加个",
            "添加",
            "加入",
            "拿着",
            "手持",
            "放一个",
            "抱着",
            "戴上",
        ),
    },
    {
        "name": "style",
        "keywords": (
            "风格化",
            "水彩风格",
            "油画风格",
            "像素风格",
            "水墨风格",
            "改成赛博",
            "画成赛博",
        ),
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
    text = _clean_detail(user_prompt)
    if not text or not re.search(r"[\u4e00-\u9fff]", text):
        return None
    if re.search(r"不要|别|禁止|不许|不能|保持|保留|无需|不改变|不修改", text):
        return None
    if re.search(r"[，,；;。]|同时|并且|然后|以及", text):
        return None
    matches = [
        template
        for template in _TEMPLATES
        if any(
            keyword in text
            for keyword in template["keywords"]
            if isinstance(keyword, str)
        )
    ]
    if len(matches) != 1:
        return None
    template = matches[0]
    name = str(template["name"])
    count = max(1, int(image_count or 1))
    if count > 1 and name != "outfit":
        return None
    if name in {"background", "expression", "hair", "style"} and not re.search(
        r"换|改|变|画成|风格化|染", text
    ):
        return None
    detail = _extract_change(text)
    if detail == text and name in {"background", "expression", "hair", "style"}:
        return None
    if (
        detail == text
        and name == "prop"
        and not any(action in text for action in ("拿着", "手持", "抱着", "戴上"))
    ):
        return None
    if detail == text and text in {
        "换装",
        "换衣服",
        "换背景",
        "换场景",
        "换表情",
        "换姿势",
        "换姿态",
        "改发色",
        "风格化",
    }:
        if name != "outfit" or count == 1:
            return None
    if name == "outfit" and count > 1:
        detail = _strip_mention_tokens(detail)
        if detail == text and len(detail) <= 6:
            detail = ""
        if detail and detail not in _MENTION_ONLY_LEFT:
            return (
                name,
                f"{TWO_IMAGE_OUTFIT_PROMPT} Additional outfit detail: {detail}.",
            )
        return name, TWO_IMAGE_OUTFIT_PROMPT
    chinese = {
        "outfit": f"将输入图中人物的服装换成{detail}。保留人物身份、姿势、构图、背景及原有画风，让新服装自然贴合身体。",
        "expression": f"将输入图中人物的表情改为{detail}，保持身份及其他未指定内容不变。",
        "background": f"把输入图的背景替换为{detail}，保留前景主体及原有画风，让新环境的光照与主体协调。",
        "pose": f"让输入图中的人物{detail}，保留身份和服装，使身体结构自然。",
        "hair": f"将输入图中人物的发色改为{detail}，保留发型、身份及其他未指定内容。",
        "prop": (
            f"让输入图中的人物{re.sub(r'^让(?:她|他|人物|角色)', '', text)}，保持人物身份及其他未指定内容不变。"
            if any(action in text for action in ("拿着", "手持", "抱着", "戴上"))
            else f"在输入图中添加{detail}，保持主体及其他内容不变，让物件的尺度与接触关系自然。"
        ),
        "style": f"将输入图整体改为{detail}风格，保留主体内容与构图。",
    }
    return name, chinese[name]


def template_names() -> list[str]:
    """List available fixed template names.

    Returns:
        Template names in match-priority order.
    """
    return [str(template["name"]) for template in _TEMPLATES]
