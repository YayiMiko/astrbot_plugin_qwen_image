"""Ground image-edit requests in the shared Qwen-Image-2.1 edit skill."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from .prompt_templates import TWO_IMAGE_OUTFIT_PROMPT, match_template
except ImportError:  # pragma: no cover - fallback for direct script-style imports.
    from prompt_templates import TWO_IMAGE_OUTFIT_PROMPT, match_template

QWEN_REWRITE_SYSTEM_PROMPT = (
    (
        Path(__file__).resolve().parent
        / "skills"
        / "qwen-image-21-prompt-expert"
        / "references"
        / "edit-policy.md"
    )
    .read_text(encoding="utf-8")
    .strip()
)

EDIT_FALLBACK_SKELETON = (
    "Edit the supplied target image as requested: {change}. Preserve its identity, original "
    "visual medium, and all content not targeted by this change."
)

RAW_PREFIXES = ("原样", "无优化", "raw:")


@dataclass
class PromptPipelineResult:
    """Final prompt plus a serializable summary dict."""

    final_prompt: str
    summary: dict[str, Any] = field(default_factory=dict)


def strip_raw_prefix(prompt: str) -> tuple[bool, str]:
    """Split an explicit raw/passthrough prefix from the prompt.

    Args:
        prompt: User prompt text.

    Returns:
        Tuple of matched flag and remaining prompt text.
    """
    text = str(prompt or "").strip()
    lowered = text.lower()
    for prefix in RAW_PREFIXES:
        if lowered.startswith(prefix.lower()):
            return True, text[len(prefix) :].strip(" ，,：:")
    return False, text


class PromptPipeline:
    """Rewrite user requests into Qwen-Image-2.1 edit paragraphs."""

    def __init__(
        self,
        *,
        context: Any,
        config: dict[str, Any],
        logger: Any,
        get_bool: Any,
        get_int: Any,
        get_str: Any,
        shorten: Any,
    ):
        """Store rewriting dependencies.

        Args:
            context: AstrBot plugin context for LLM calls.
            config: Plugin configuration dict.
            logger: Logger compatible with AstrBot logger methods.
            get_bool: Config boolean accessor.
            get_int: Config integer accessor.
            get_str: Config string accessor.
            shorten: Text-shortening helper.
        """
        self.context = context
        self.config = config
        self.logger = logger
        self._bool = get_bool
        self._int = get_int
        self._str = get_str
        self._shorten = shorten

    async def _current_chat_provider_id(self, event: Any) -> str:
        configured = self._str("prompt_builder_provider_id", "").strip()
        if configured:
            return configured
        try:
            provider_id = await self.context.get_current_chat_provider_id(
                event.unified_msg_origin
            )
            if str(provider_id or "").strip():
                return str(provider_id).strip()
            cfg = self.context.get_config(umo=event.unified_msg_origin)
            return str(
                cfg.get("provider_settings", {}).get("default_provider_id") or ""
            ).strip()
        except Exception as exc:
            self.logger.warning("[qwen] failed to get current chat provider: %s", exc)
            return ""

    async def _rewrite_with_llm(
        self,
        *,
        provider_id: str,
        user_prompt: str,
        image_count: int,
        system_prompt: str,
        image_urls: list[str] | None = None,
        vision_outfit: bool = False,
    ) -> str:
        llm_prompt = (
            f"Edit request from the chat user:\n{user_prompt}\n\n"
            f"Supplied images: {max(1, image_count)} "
            "(first image is <image1>, the edit target; "
            "others are <image2>/<image3> references)."
        )
        if vision_outfit and image_urls:
            llm_prompt += (
                "\nLook at the attached images: the FIRST image is the edit "
                "target (<image1>), the SECOND image is the outfit reference "
                "(<image2>). Name the distinctive garments and details you "
                "can actually see when they help a faithful transfer; do not "
                "invent hidden parts, materials, or accessories."
            )
            llm_prompt += (
                "\nUse this validated two-image outfit prompt as the backbone, "
                "adding only the user's requested details and the reference "
                f"garment details you can actually see:\n{TWO_IMAGE_OUTFIT_PROMPT}"
            )
        response = await self.context.llm_generate(
            chat_provider_id=provider_id,
            prompt=llm_prompt,
            system_prompt=system_prompt,
            max_tokens=self._int("prompt_builder_max_tokens", 800),
            image_urls=list(image_urls or []) or None,
        )
        return str(getattr(response, "completion_text", "") or "").strip()

    async def build(
        self,
        event: Any,
        user_prompt: str,
        mode: str = "img2img",
        image_count: int = 1,
        image_paths: list[str] | None = None,
    ) -> PromptPipelineResult:
        """Build the final edit paragraph and summary for one request.

        Args:
            event: AstrBot message event for provider lookup.
            user_prompt: Raw user requirement text.
            mode: Generation mode (only `img2img` is supported).
            image_count: Number of supplied images (1-3).
            image_paths: Saved local image paths, passed to a vision-capable
                rewriter so reference garments can be described explicitly.

        Returns:
            Final prompt plus a serializable summary dict.
        """
        prompt = str(user_prompt or "").strip()
        summary: dict[str, Any] = {
            "prompt_optimize_enabled": self._bool("prompt_optimize_enabled", True),
            "mode": mode,
            "image_count": max(1, int(image_count or 1)),
            "original_prompt_head": self._shorten(prompt, 600),
        }
        if mode != "img2img":
            summary.update(
                {
                    "skipped_reason": "unsupported_mode",
                    "final_prompt_head": self._shorten(prompt, 600),
                    "final_prompt_chars": len(prompt),
                }
            )
            return PromptPipelineResult(prompt, summary)
        raw_mode, raw_prompt = strip_raw_prefix(prompt)
        if raw_mode:
            self.logger.info("[qwen] prompt rewrite skipped: raw mode")
            summary.update(
                {
                    "raw_mode": True,
                    "skipped_reason": "raw_mode",
                    "final_prompt_head": self._shorten(raw_prompt, 600),
                    "final_prompt_chars": len(raw_prompt),
                }
            )
            return PromptPipelineResult(raw_prompt, summary)
        if not self._bool("prompt_optimize_enabled", True):
            fallback = EDIT_FALLBACK_SKELETON.format(change=prompt or "redraw")
            summary.update(
                {
                    "skipped_reason": "prompt_optimize_disabled",
                    "tier": "fallback",
                    "llm_used": False,
                    "final_prompt_head": self._shorten(fallback, 600),
                    "final_prompt_chars": len(fallback),
                }
            )
            return PromptPipelineResult(fallback, summary)
        prompt_mode = self._str("prompt_mode", "auto").strip() or "auto"
        if prompt_mode not in ("auto", "template_only", "llm_only"):
            prompt_mode = "auto"
        summary["prompt_mode"] = prompt_mode
        paths = [str(path) for path in (image_paths or []) if str(path).strip()]
        matched = (
            match_template(prompt, summary["image_count"])
            if prompt_mode != "llm_only"
            else None
        )
        # Outfit transfers with references go through a vision-grounded
        # rewrite: the model must SEE the garment to name it explicitly,
        # otherwise only the colour bleeds through. Falls back to the fixed
        # template below when the vision call fails.
        vision_outfit = (
            prompt_mode == "auto"
            and matched is not None
            and matched[0] == "outfit"
            and len(paths) > 1
        )
        if matched is not None and not vision_outfit:
            template_name, templated = matched
            self.logger.info("[qwen] fixed template hit: %s", template_name)
            summary.update(
                {
                    "tier": f"template:{template_name}",
                    "llm_used": False,
                    "final_prompt_head": self._shorten(templated, 600),
                    "final_prompt_chars": len(templated),
                }
            )
            return PromptPipelineResult(templated, summary)
        if prompt_mode == "template_only":
            fallback = EDIT_FALLBACK_SKELETON.format(change=prompt or "redraw")
            summary.update(
                {
                    "skipped_reason": "template_miss",
                    "tier": "fallback",
                    "llm_used": False,
                    "final_prompt_head": self._shorten(fallback, 600),
                    "final_prompt_chars": len(fallback),
                }
            )
            return PromptPipelineResult(fallback, summary)
        provider_id = await self._current_chat_provider_id(event)
        summary["provider_id"] = provider_id
        summary["vision_images"] = len(paths) if paths else 0
        system_prompt = (
            self._str("prompt_rewrite_system_prompt", "").strip()
            or QWEN_REWRITE_SYSTEM_PROMPT
        )
        try:
            rewritten = await self._rewrite_with_llm(
                provider_id=provider_id,
                user_prompt=prompt,
                image_count=summary["image_count"],
                system_prompt=system_prompt,
                image_urls=paths or None,
                vision_outfit=vision_outfit,
            )
        except Exception as exc:
            self.logger.warning("[qwen] prompt rewrite failed: %s", exc)
            rewritten = ""
        if not rewritten:
            if matched is not None:
                template_name, templated = matched
                summary.update(
                    {
                        "skipped_reason": "rewrite_failed",
                        "tier": f"template:{template_name}",
                        "llm_used": False,
                        "final_prompt_head": self._shorten(templated, 600),
                        "final_prompt_chars": len(templated),
                    }
                )
                return PromptPipelineResult(templated, summary)
            rewritten = EDIT_FALLBACK_SKELETON.format(change=prompt or "redraw")
            summary.update(
                {
                    "llm_used": False,
                    "tier": "fallback",
                    "skipped_reason": "rewrite_failed",
                    "final_prompt_head": self._shorten(rewritten, 600),
                    "final_prompt_chars": len(rewritten),
                }
            )
            return PromptPipelineResult(rewritten, summary)
        summary.update(
            {
                "llm_used": True,
                "tier": "llm:vision" if vision_outfit else "llm",
                "final_prompt_head": self._shorten(rewritten, 600),
                "final_prompt_chars": len(rewritten),
            }
        )
        return PromptPipelineResult(rewritten, summary)
