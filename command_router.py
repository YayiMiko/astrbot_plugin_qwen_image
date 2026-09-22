import re

_ROUTE_PREFIX_RE = re.compile(r"^\s*/?", re.IGNORECASE)
_SPACES_RE = re.compile(r"\s+")


def help_text(img2img_enabled: bool = False) -> str:
    """Build the chat-visible Qwen command help text.

    Args:
        img2img_enabled: Whether to show the img2img/edit command.

    Returns:
        The help text shown in chat.
    """
    lines = [
        "Qwen 指令表：",
        "- /qwen 状态：查看 ComfyUI / Qwen-Image-2.1 状态",
        "- /qwen 生图 <描述>：文生图正在开发中，暂不可用",
        "- /qwen 记图：把本条带的图/引用的图记为参考图1/2/3（最多 3 张，30 分钟有效）",
        "- /qwen 看图：查看已标记的参考图",
        "- /qwen 清图：清空已标记的参考图",
    ]
    if img2img_enabled:
        lines.extend(
            [
                "- /qwen 改图（编辑）<要求>：按要求编辑图片，改图和编辑是同一个功能",
                "  第一张为改图目标，后面为参考图（换装/融合/风格参考），最多 3 张",
                "  点名目标会自动排第一：/qwen 编辑 为图2角色穿上图1的衣服",
                "  输出尺寸跟随目标图；大图会自动缩到最长边 1024（配置可改）",
            ]
        )
    lines.extend(["", "例：/qwen 改图 让图一的角色穿上图二的衣服"])
    return "\n".join(lines)


def normalize_route_text(text: str) -> str:
    """Normalize a command-like chat message.

    Args:
        text: Raw chat text.

    Returns:
        Text with the optional leading slash and repeated spaces removed.
    """
    text = _ROUTE_PREFIX_RE.sub("", str(text or "")).strip()
    return _SPACES_RE.sub(" ", text)


def parse_hard_route(text: str) -> tuple[str, str] | None:
    """Parse a Qwen hard-route command.

    Args:
        text: Raw chat text or message outline.

    Returns:
        A tuple of action and prompt when the message should be handled by
        Qwen, otherwise None.
    """
    raw_text = str(text or "")
    explicit_command = bool(re.match(r"^\s*/", raw_text))
    normalized = normalize_route_text(raw_text)
    lowered = normalized.lower()
    prefixes = ("qwen",)

    for prefix in prefixes:
        if not explicit_command:
            break
        if not lowered.startswith(prefix.lower()):
            continue
        rest = normalized[len(prefix) :].strip(" ，,：:")
        if not rest:
            return "help", ""
        rest_lower = rest.lower()
        action_map = [
            ("help", "help"),
            ("帮助", "help"),
            ("指令表", "help"),
            ("指令", "help"),
            ("菜单", "help"),
            ("status", "status"),
            ("状态", "status"),
            ("diagnose", "diagnose"),
            ("diagnosis", "diagnose"),
            ("诊断", "diagnose"),
            ("部署诊断", "diagnose"),
            ("debug_status", "debug_status"),
            ("debug", "debug_status"),
            ("调试状态", "debug_status"),
            ("调试", "debug_status"),
            ("generate", "t2i_stub"),
            ("生图", "t2i_stub"),
            ("画图", "t2i_stub"),
            ("文生图", "t2i_stub"),
            ("edit", "edit"),
            ("编辑", "edit"),
            ("改编", "edit"),
            ("改图", "edit"),
            ("mark_slots", "mark_slots"),
            ("记图", "mark_slots"),
            ("标记", "mark_slots"),
            ("存图", "mark_slots"),
            ("show_slots", "show_slots"),
            ("看图", "show_slots"),
            ("槽位", "show_slots"),
            ("查看标记", "show_slots"),
            ("clear_slots", "clear_slots"),
            ("清图", "clear_slots"),
            ("清除标记", "clear_slots"),
            ("删除标记", "clear_slots"),
            ("图生图", "edit"),
            ("换装", "edit"),
            ("融合", "edit"),
            ("风格化", "edit"),
            ("重绘", "edit"),
        ]
        for keyword, action in action_map:
            if not rest_lower.startswith(keyword.lower()):
                continue
            prompt = rest[len(keyword) :].strip(" ，,：:")
            return action, prompt
        # Bare "/qwen <text>" with an image attached is treated as an edit
        # request; without an image it falls through to the T2I stub.
        return "edit", rest

    natural = re.match(
        r"^(?:用\s*)?qwen"
        r"(?:帮我|给我|来)?"
        r"\s*"
        r"(帮助|指令表|指令|菜单|help|状态|status|诊断|部署诊断|diagnose|diagnosis|调试状态|调试|debug_status|debug|记图|标记|存图|看图|槽位|清图|清除标记|改图|编辑|改编|换装|融合|重绘|风格化|生图|画图|文生图|生成)"
        r"\s*(.*)$",
        normalized,
        flags=re.IGNORECASE,
    )
    if not natural:
        return None
    verb = natural.group(1)
    prompt = natural.group(2).strip(" ，,：:")
    if verb.lower() == "help" or verb in {"帮助", "指令表", "指令", "菜单"}:
        return "help", prompt
    if verb.lower() == "status" or verb == "状态":
        return "status", prompt
    if verb.lower() in {"diagnose", "diagnosis"} or verb in {"诊断", "部署诊断"}:
        return "diagnose", prompt
    if verb.lower() in {"debug_status", "debug"} or verb in {"调试状态", "调试"}:
        return "debug_status", prompt
    if verb in {"改图", "编辑", "改编", "换装", "融合", "重绘", "风格化"}:
        return "edit", prompt
    if verb in {"记图", "标记", "存图"}:
        return "mark_slots", prompt
    if verb in {"看图", "槽位"}:
        return "show_slots", prompt
    if verb in {"清图", "清除标记"}:
        return "clear_slots", prompt
    return "t2i_stub", prompt
