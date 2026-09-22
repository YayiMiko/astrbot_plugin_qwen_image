from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .agent_tools.comfyui_workflows import MAX_EDIT_IMAGES
    from .command_router import help_text
    from .config_defaults import persist_flat_config_key
    from .deployment_diagnostics import compact_status_text, diagnostic_text
    from .image_slots import (
        find_image_mentions,
        reorder_target_first,
        sanitize_session_key,
        slot_key,
    )
except Exception:  # pragma: no cover - fallback for direct script-style imports.
    from agent_tools.comfyui_workflows import MAX_EDIT_IMAGES
    from command_router import help_text
    from config_defaults import persist_flat_config_key
    from deployment_diagnostics import compact_status_text, diagnostic_text
    from image_slots import (
        find_image_mentions,
        reorder_target_first,
        sanitize_session_key,
        slot_key,
    )


SCHEMA_PATH = Path(__file__).with_name("_conf_schema.json")

T2I_UNAVAILABLE = (
    "Qwen 文生图正在开发中，暂不可用。当前可用 /qwen 改图（图生图/换装/多图融合）。"
)

PROBE_PROMPT = (
    "Keep the character and pose in <image1> unchanged, replace the "
    "background with a plain light-gray studio backdrop, preserve the "
    "original facial features, hair, body shape and pose, keep the original "
    "lighting, sharp details"
)


class CommandActionHandler:
    """Dispatch chat commands to Qwen plugin actions."""

    def __init__(
        self,
        *,
        config: dict[str, Any],
        task_recorder: Any,
        is_allowed: Callable[[Any], bool],
        run_tool: Callable[[list[str]], Any],
        ensure_ready: Callable[[Any], Any],
        send_payload: Callable[[Any, dict[str, Any]], Any],
        event_image_inputs: Callable[[Any, int], Any],
        image_input_summary: Callable[[], dict[str, Any]],
        slot_store: Any,
        prepare_probe_images: Callable[[], list[str]],
        build_prompt: Callable[..., Any],
        prompt_summary: Callable[[], dict[str, Any]],
        get_bool: Callable[[str, bool], bool],
        shorten: Callable[[str, int], str],
        config_store: Any = None,
    ):
        """Store dependencies for command-side action handling.

        Args:
            config: Plugin configuration dict.
            task_recorder: Task recorder used for debug status output.
            is_allowed: Permission checker.
            run_tool: Main ComfyUI helper runner.
            ensure_ready: ComfyUI readiness checker.
            send_payload: Chat payload sender.
            event_image_inputs: Multi-image input resolver.
            image_input_summary: Latest image input summary callback.
            build_prompt: Prompt rewriter for img2img.
            get_bool: Config boolean accessor.
            shorten: Text-shortening helper.
        """
        self.config = config
        self._config_store = config_store
        self._task_recorder = task_recorder
        self._is_allowed = is_allowed
        self._run_tool = run_tool
        self._ensure_ready = ensure_ready
        self._send_payload = send_payload
        self._event_image_inputs = event_image_inputs
        self._image_input_summary = image_input_summary
        self._slot_store = slot_store
        self._prepare_probe_images = prepare_probe_images
        self._build_prompt = build_prompt
        self._prompt_summary = prompt_summary
        self._bool = get_bool
        self._shorten = shorten

    def _persist_config_key(self, key: str, value: Any) -> None:
        self.config[key] = value
        if isinstance(self._config_store, dict):
            persist_flat_config_key(self._config_store, SCHEMA_PATH, key, value)
        save_config = getattr(self._config_store, "save_config", None)
        if callable(save_config):
            save_config()

    def status_text(self, payload: dict[str, Any]) -> str:
        """Render a chat-visible ComfyUI status response.

        Args:
            payload: Status payload returned by the ComfyUI helper.

        Returns:
            Human-readable status text.
        """
        return compact_status_text(payload)

    def diagnose_text(self, payload: dict[str, Any]) -> str:
        """Render a deployment-focused diagnostic response.

        Args:
            payload: Status payload returned by the ComfyUI helper.

        Returns:
            Human-readable diagnostic text.
        """
        return diagnostic_text(
            payload,
            self.config,
            self._task_recorder.read(),
            self._image_input_summary(),
        )

    def _slot_key(self, event: Any) -> str:
        """Build the slot key for an event.

        Scope comes from config: `group` shares one slot set per session
        (group chat default, preserves cross-user flows), `user` keeps an
        independent set per sender.

        Args:
            event: AstrBot message event.

        Returns:
            Slot key string.
        """
        try:
            session = str(event.get_session_id() or "default")
        except Exception:
            session = "default"
        scope = str(self.config.get("slot_scope") or "group").strip().lower()
        if scope != "user":
            return sanitize_session_key(session)
        try:
            sender = str(event.get_sender_id() or "unknown")
        except Exception:
            sender = "unknown"
        return slot_key(session, sender)

    async def mark_slots(self, event: Any) -> str:
        """Stage attached/quoted images into the session's reference slots."""
        if not self._is_allowed(event):
            return "Qwen 助手已关闭，或当前用户没有使用权限。"
        image_paths = await self._event_image_inputs(event, MAX_EDIT_IMAGES)
        if not image_paths:
            return (
                "没有找到可标记的图片。请把图和指令发在同一条消息里，"
                "或引用一条包含图片的消息再 /qwen 记图。"
            )
        before = {
            str(slot.get("path"))
            for slot in self._slot_store.read(self._slot_key(event))
        }
        result = self._slot_store.save(self._slot_key(event), image_paths)
        slots = result["slots"]
        new_indices = [
            index
            for index, slot in enumerate(slots, start=1)
            if str(slot.get("path")) not in before
        ]
        if not new_indices:
            names = "、".join(
                f"图{index}"
                for index, slot in enumerate(slots, start=1)
                if str(slot.get("path")) in {str(path) for path in image_paths}
            ) or "已标记的图"
            return (
                f"这几张已经标记过了（{names}），不用重发。"
                "用 /qwen 看图 查看全部已标记参考图。"
            )
        lines = [
            f"本次新增 {len(new_indices)} 张"
            f"（{', '.join(f'图{i}' for i in new_indices)}），"
            f"当前共 {len(slots)} 张参考图。"
        ]
        if result["evicted"]:
            lines.append(f"槽位已满，最早的 {result['evicted']} 张已被顶掉。")
        lines.append("接下来可以直接说：/qwen 编辑 为图1角色穿上图二的衣服")
        return "\n".join(lines)

    def show_slots(self, event: Any) -> str:
        """List the session's staged reference slots."""
        slots = self._slot_store.read(self._slot_key(event))
        if not slots:
            return "当前没有已标记的参考图。用 /qwen 记图 先标记（附图或引用图片）。"
        lines = [f"已标记 {len(slots)} 张参考图："]
        for index, slot in enumerate(slots, start=1):
            name = Path(str(slot.get("path") or "")).name
            lines.append(f"- 图{index}：{name or '未知文件'}")
        lines.append("用法：/qwen 编辑 为图1角色穿上图二的衣服；/qwen 清图 可清空。")
        return "\n".join(lines)

    def clear_slots(self, event: Any) -> str:
        """Clear the session's staged reference slots."""
        removed = self._slot_store.clear(self._slot_key(event))
        if not removed:
            return "当前没有已标记的参考图，无需清理。"
        return f"已清理 {removed} 张已标记参考图。"

    def _resolve_slot_images(
        self, event: Any, prompt: str
    ) -> tuple[list[str], list[int], str | None]:
        """Resolve slot images for mentions like 图一/图二, or all slots.

        Args:
            event: AstrBot message event.
            prompt: User requirement text.

        Returns:
            Tuple of image paths, used 1-based slot numbers, and an optional
            error message (empty paths + error means failure).
        """
        slots = self._slot_store.read(self._slot_key(event))
        if not slots:
            return [], [], (
                "提到了图一/图二，但当前没有已标记的参考图。"
                "先用 /qwen 记图 标记（附图或引用图片），或用 /qwen 看图 确认。"
            )
        mentions = [n for n in find_image_mentions(prompt) if 1 <= n <= MAX_EDIT_IMAGES]
        if not mentions:
            mentions = list(range(1, len(slots) + 1))
        missing = [n for n in mentions if n > len(slots)]
        if missing:
            return [], [], (
                f"图{missing[0]}没有标记（当前只有 {len(slots)} 张）。"
                "先用 /qwen 记图 补标记，或用 /qwen 看图 确认顺序。"
            )
        paths = [str(slots[n - 1].get("path")) for n in mentions]
        return paths, mentions, None

    async def edit(self, event: Any, prompt: str) -> str:
        if not self._bool("img2img_enabled", True):
            return "图生图/改图功能已关闭，请在插件配置里开启后再试。"
        if not self._is_allowed(event):
            return "Qwen 助手已关闭，或当前用户没有使用权限。"
        ready = await self._ensure_ready(event)
        if not ready.get("ok"):
            return await self._send_payload(event, ready)
        prompt = str(prompt or "").strip()
        if not prompt:
            return "请在后面写改图要求。例：/qwen 改图 让图一的角色穿上图二的衣服"
        raw_prompt = prompt
        image_paths = await self._event_image_inputs(event, MAX_EDIT_IMAGES)
        image_source = "attached"
        used_slots: list[int] = []
        if not image_paths:
            image_paths, used_slots, slot_error = self._resolve_slot_images(
                event, prompt
            )
            if slot_error:
                if find_image_mentions(prompt):
                    return slot_error
                return (
                    "没有拿到参考图。请把图和指令发在同一条消息里，引用一条包含图片的消息，"
                    "或先用 /qwen 记图 标记参考图后再试。"
                    "例如：/qwen 改图 让图一的角色穿上图二的衣服（附两张图）。"
                )
            image_source = "slots"
        slot_numbers = (
            used_slots
            if image_source == "slots"
            else list(range(1, len(image_paths) + 1))
        )
        image_paths, slot_numbers, prompt = reorder_target_first(
            image_paths, slot_numbers, prompt
        )
        used_slots = slot_numbers if image_source == "slots" else []
        original = raw_prompt
        if self._bool("prompt_optimize_enabled", True):
            prompt = await self._build_prompt(
                event,
                prompt,
                mode="img2img",
                image_count=len(image_paths),
                image_paths=image_paths,
            )
        tool_args = ["edit", "--prompt", prompt]
        for image_path in image_paths:
            tool_args.extend(["--input", str(image_path)])
        payload = await self._run_tool(tool_args)
        message = await self._send_payload(event, payload)
        self._record_edit_task(event, original, image_paths, payload, image_source)
        if image_source == "slots" and payload.get("ok") and used_slots:
            names = "、".join(f"图{n}" for n in used_slots)
            message += f"（本次使用已标记的{names}。）"
        truncated = int(payload.get("truncated_inputs") or 0)
        if truncated > 0 and payload.get("ok"):
            message += (
                f"（Qwen 编辑节点最多接受 {MAX_EDIT_IMAGES} 张参考图，"
                f"已取前 {MAX_EDIT_IMAGES} 张，其余 {truncated} 张未使用。）"
            )
        return message

    def _record_edit_task(
        self,
        event: Any,
        original: str,
        image_paths: list[str],
        payload: dict[str, Any],
        image_source: str = "attached",
    ) -> None:
        """Persist a non-secret summary of the finished edit task.

        Args:
            event: AstrBot message event.
            original: User requirement before rewriting.
            image_paths: Saved local input image paths.
            payload: ComfyUI helper result payload.
            image_source: Where the images came from (`attached` or `slots`).
        """
        try:
            summary = dict(self._prompt_summary() or {})
        except Exception:
            summary = {}
        task = {
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "action": "edit",
            "platform_id": event.get_platform_id(),
            "session_id": event.get_session_id(),
            "sender_id": event.get_sender_id(),
            "ok": bool(payload.get("ok")),
            "error": payload.get("error") or "",
            "workflow": self.config.get("workflow", "qwen21_edit"),
            "inputs": [str(path) for path in image_paths],
            "image_source": image_source,
            "parameters": {
                "steps": payload.get("steps"),
                "cfg": payload.get("cfg"),
                "seed": payload.get("seed"),
            },
            "prompt": {"original_head": self._shorten(original, 1000)},
            "prompt_summary": summary,
            "outputs": list(payload.get("outputs") or []),
            "delivery": dict(payload.get("delivery") or {}),
        }
        try:
            self._task_recorder.write(task)
        except Exception:
            pass

    def _check_records(self) -> list[Path]:
        parent = self._task_recorder.path.parent
        try:
            records = sorted(
                parent.glob("workflow_check_*.json"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        except OSError:
            records = []
        return records

    async def check_workflow(self, event: Any) -> str:
        """Validate the custom workflow with shipped probe images."""
        if not self._is_allowed(event):
            return "Qwen 助手已关闭，或当前用户没有使用权限。"
        if not self._bool("custom_workflow_enabled", False):
            return (
                "当前使用内置工作流，无需校验。"
                "在插件配置里开启“使用自定义工作流”并填写路径后再 /qwen 验工作流。"
            )
        ready = await self._ensure_ready(event)
        if not ready.get("ok"):
            return await self._send_payload(event, ready)
        staged = self._prepare_probe_images()
        if not staged:
            return "探针图片缺失，无法校验。请检查插件目录 probe/ 是否完整。"
        tool_args = ["edit", "--prompt", PROBE_PROMPT, "--steps", "8"]
        for image_path in staged[:MAX_EDIT_IMAGES]:
            tool_args.extend(["--input", str(image_path)])
        payload = await self._run_tool(tool_args)
        record = {
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "action": "check_workflow",
            "ok": bool(payload.get("ok")),
            "error": payload.get("error") or "",
            "prompt_id": payload.get("prompt_id") or "",
            "status": payload.get("status") or {},
            "outputs": list(payload.get("outputs") or []),
        }
        record_name = (
            "workflow_check_"
            + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
            + ".json"
        )
        try:
            parent = self._task_recorder.path.parent
            parent.mkdir(parents=True, exist_ok=True)
            import json

            (parent / record_name).write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            record_name = ""
        if payload.get("ok"):
            return (
                "自定义工作流校验通过："
                f"探针 8 步出图 {len(record['outputs'])} 张。"
                + (f"记录：{record_name}" if record_name else "")
            )
        reason = str(payload.get("error") or "unknown_error")
        return (
            f"自定义工作流校验失败：{reason}。"
            + (f"完整日志已保存：{record_name}，" if record_name else "")
            + "用 /qwen 取日志 发出来，我帮你看看怎么修。"
        )

    def fetch_check_log(self, event: Any, prompt: str) -> str:
        """Return a saved workflow-check record as copyable text."""
        _ = event
        records = self._check_records()
        if not records:
            return "还没有工作流校验记录。先用 /qwen 验工作流 跑一次。"
        index = 0
        for token in str(prompt or "").split():
            if token.isdigit():
                index = max(0, int(token) - 1)
                break
        if index >= len(records):
            return f"只有 {len(records)} 条校验记录，没有第 {index + 1} 条。"
        try:
            import json

            record = json.loads(records[index].read_text(encoding="utf-8"))
        except Exception:
            return "读取校验记录失败，文件可能已损坏。"
        text = json.dumps(record, ensure_ascii=False, indent=2)
        return (
            f"工作流校验记录（{records[index].name}，共 {len(records)} 条）：\n"
            + self._shorten(text, 3000)
        )

    async def handle_action(self, event: Any, action: str, prompt: str) -> str | None:
        """Handle one parsed command action.

        Args:
            event: AstrBot message event.
            action: Parsed action key from `command_router`.
            prompt: Prompt text after the action keyword.

        Returns:
            Optional message to send back.
        """
        if not self._is_allowed(event):
            return "Qwen 助手已关闭，或当前用户没有使用权限。"
        if action == "help":
            return help_text(self._bool("img2img_enabled", True))
        if action == "status":
            return self.status_text(await self._run_tool(["status"]))
        if action == "diagnose":
            return self.diagnose_text(await self._run_tool(["status"]))
        if action == "debug_status":
            return self._task_recorder.debug_status_text(self.config)
        if action == "t2i_stub":
            return T2I_UNAVAILABLE
        if action == "mark_slots":
            return await self.mark_slots(event)
        if action == "show_slots":
            return self.show_slots(event)
        if action == "clear_slots":
            return self.clear_slots(event)
        if action == "check_workflow":
            return await self.check_workflow(event)
        if action == "fetch_check_log":
            return self.fetch_check_log(event, prompt)
        if action == "edit":
            if not prompt:
                return "请在后面写改图要求。例：/qwen 改图 让图一的角色穿上图二的衣服"
            return await self.edit(event, prompt)
        return "未知 Qwen 指令。"
