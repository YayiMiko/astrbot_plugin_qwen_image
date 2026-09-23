from __future__ import annotations

from collections.abc import Callable
from typing import Any


class LLMToolBridge:
    """Bridge AstrBot LLM tool methods to Qwen plugin services."""

    def __init__(
        self,
        *,
        run_tool: Callable[[list[str]], Any],
        edit: Callable[[Any, str], Any],
    ):
        """Store callbacks used by decorated LLM tool entry points.

        Args:
            run_tool: Main ComfyUI helper runner.
            edit: Image edit callback.
        """
        self._run_tool = run_tool
        self._edit = edit

    async def status(self, event: Any) -> str:
        """Return a compact status string for LLM tool use.

        Args:
            event: AstrBot message event.

        Returns:
            Compact English status text for the model.
        """
        payload = await self._run_tool(["status"])
        if not payload.get("ok"):
            return f"ComfyUI status failed: {payload.get('error')}"
        return (
            "ComfyUI status: "
            f"base_url={payload.get('base_url')}, "
            f"workflow={payload.get('workflow')}, "
            f"version={payload.get('comfyui_version')}, "
            f"gpu={payload.get('gpu')}, "
            f"vram_free_mb={payload.get('vram_free_mb')}, "
            f"unet_available={payload.get('unet_available')}, "
            f"clip_available={payload.get('clip_available')}, "
            f"vae_available={payload.get('vae_available')}, "
            f"qwen_edit_available={payload.get('qwen_edit_available')}. "
            "Text-to-image is NOT implemented yet; only reference editing "
            "(1-3 images, first is the edit target) is supported."
        )

    async def edit(self, event: Any, prompt: str) -> str:
        """Edit the most recent or quoted images.

        Args:
            event: AstrBot message event.
            prompt: Edit requirement in any language.

        Returns:
            Edit result summary.
        """
        result = await self._edit(event, prompt)
        return (
            result
            or "The image or error message was already sent to chat. Do not repeat it."
        )
