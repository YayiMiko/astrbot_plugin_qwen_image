import sys
from pathlib import Path
from typing import Any

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.event.filter import EventMessageType
from astrbot.api.star import Context, Star
from astrbot.core.star.filter.command import GreedyStr

try:
    from .command_router import parse_hard_route
    from .config_defaults import (
        flatten_config,
        group_config,
        maybe_migrate_to_grouped_config,
        maybe_reset_to_defaults,
    )
    from .service_container import build_services
except Exception:  # pragma: no cover - fallback for direct script-style imports.
    from command_router import parse_hard_route
    from config_defaults import (
        flatten_config,
        group_config,
        maybe_migrate_to_grouped_config,
        maybe_reset_to_defaults,
    )
    from service_container import build_services


class QwenImagePlugin(Star):
    """Qwen-Image-2.1 reference editing plugin for AstrBot."""

    def __init__(self, context: Context, config: AstrBotConfig | None = None):
        super().__init__(context, config)
        schema_path = Path(__file__).with_name("_conf_schema.json")
        grouped_or_reset = maybe_migrate_to_grouped_config(config or {}, schema_path)
        raw_or_reset = maybe_reset_to_defaults(
            config or {},
            schema_path,
        )
        raw_config = flatten_config(raw_or_reset or grouped_or_reset)
        if config is not None and group_config(raw_config, schema_path) != dict(
            config or {}
        ):
            config.save_config(replace_config=group_config(raw_config, schema_path))
        self.config = raw_config
        self._services = build_services(
            context=self.context,
            config=self.config,
            config_store=config,
            logger=logger,
            get_bool=self._bool,
            get_int=self._int,
            get_str=self._str,
            shorten=self._shorten,
            is_allowed=self._is_allowed,
            edit=self._edit,
        )
        self._runtime = self._services.runtime
        self._action_handler = self._services.action_handler
        self._llm_tool_bridge = self._services.llm_tool_bridge

    async def initialize(self):
        img2img_enabled = self._bool("img2img_enabled", True)
        if img2img_enabled:
            edit_tool_changed = self.context.activate_llm_tool("qwen_edit")
        else:
            edit_tool_changed = self.context.deactivate_llm_tool("qwen_edit")
        logger.info(
            "[qwen] img2img_enabled=%s edit_tool_changed=%s base_url=%s workflow=%s",
            img2img_enabled,
            edit_tool_changed,
            self._str("comfyui_base_url", "http://127.0.0.1:8188"),
            self._str("workflow", "qwen21_edit"),
        )

    def _bool(self, key: str, default: bool) -> bool:
        return bool(self.config.get(key, default))

    def _int(self, key: str, default: int) -> int:
        try:
            return int(self.config.get(key, default))
        except (TypeError, ValueError):
            return default

    def _str(self, key: str, default: str = "") -> str:
        value = self.config.get(key, default)
        return str(value if value is not None else default)

    def _is_allowed(self, event: AstrMessageEvent) -> bool:
        if self._bool("admin_only", False) and not event.is_admin():
            return False
        allowed = self.config.get("allowed_sender_ids", [])
        if isinstance(allowed, str):
            allowed = [allowed]
        allowed_set = {str(item).strip() for item in allowed or [] if str(item).strip()}
        if allowed_set and str(event.get_sender_id()) not in allowed_set:
            return False
        return True

    async def _run_tool(self, args: list[str]) -> dict[str, Any]:
        return await self._runtime.run_tool(args)

    def _shorten(self, text: str, limit: int = 1800) -> str:
        text = str(text or "").strip()
        if len(text) <= limit:
            return text
        return text[:limit].rstrip() + "\n...[已截断]"

    async def _edit(self, event: AstrMessageEvent, prompt: str) -> str | None:
        return await self._action_handler.edit(event, prompt)

    def _status_text(self, payload: dict[str, Any]) -> str:
        return self._action_handler.status_text(payload)

    async def _handle_action(
        self, event: AstrMessageEvent, action: str, prompt: str
    ) -> str | None:
        return await self._action_handler.handle_action(event, action, prompt)

    @filter.command_group("qwen")
    def qwen_group(self):
        pass

    @filter.event_message_type(EventMessageType.ALL, priority=sys.maxsize - 2)
    async def hard_route_qwen(self, event: AstrMessageEvent):
        route = parse_hard_route(event.get_message_str())
        if not route:
            route = parse_hard_route(event.get_message_outline())
        if not route:
            return
        if not self._is_allowed(event):
            await event.send(
                event.plain_result("Qwen 助手已关闭，或当前用户没有使用权限。")
            )
            event.stop_event()
            return

        action, prompt = route
        logger.info(
            "[qwen] hard route action=%s sender=%s", action, event.get_sender_id()
        )
        event.stop_event()

        message = await self._handle_action(event, action, prompt)
        if message:
            await event.send(event.plain_result(message))

    @qwen_group.command("status", alias={"状态"})
    async def cmd_status(self, event: AstrMessageEvent):
        event.stop_event()
        yield event.plain_result(await self._handle_action(event, "status", ""))

    @qwen_group.command("diagnose", alias={"诊断", "部署诊断", "diagnosis"})
    async def cmd_diagnose(self, event: AstrMessageEvent):
        event.stop_event()
        yield event.plain_result(await self._handle_action(event, "diagnose", ""))

    @qwen_group.command("debug", alias={"调试状态", "调试", "debug_status"})
    async def cmd_debug_status(self, event: AstrMessageEvent):
        event.stop_event()
        yield event.plain_result(await self._handle_action(event, "debug_status", ""))

    @qwen_group.command("generate", alias={"生图", "画图", "文生图"})
    async def cmd_generate(self, event: AstrMessageEvent, prompt: GreedyStr):
        event.stop_event()
        message = await self._handle_action(
            event, "generate", str(prompt or "").strip()
        )
        if message:
            yield event.plain_result(message)

    @qwen_group.command(
        "edit",
        alias={"编辑", "改编", "改图", "图生图", "换装", "融合", "风格化", "重绘"},
    )
    async def cmd_edit(self, event: AstrMessageEvent, prompt: GreedyStr):
        event.stop_event()
        message = await self._handle_action(event, "edit", str(prompt or "").strip())
        if message:
            yield event.plain_result(message)

    @qwen_group.command("mark", alias={"记图", "标记", "存图"})
    async def cmd_mark_slots(self, event: AstrMessageEvent):
        event.stop_event()
        yield event.plain_result(await self._handle_action(event, "mark_slots", ""))

    @qwen_group.command("slots", alias={"看图", "槽位"})
    async def cmd_show_slots(self, event: AstrMessageEvent):
        event.stop_event()
        yield event.plain_result(await self._handle_action(event, "show_slots", ""))

    @qwen_group.command("clear", alias={"清图", "清除标记"})
    async def cmd_clear_slots(self, event: AstrMessageEvent):
        event.stop_event()
        yield event.plain_result(await self._handle_action(event, "clear_slots", ""))

    @filter.llm_tool(name="qwen_status")
    async def qwen_status(self, event: AstrMessageEvent) -> str:
        """Check whether local ComfyUI with Qwen-Image-2.1 is online and ready.

        Args:
        """
        return await self._llm_tool_bridge.status(event)

    @filter.llm_tool(name="qwen_edit")
    async def qwen_edit(self, event: AstrMessageEvent, prompt: str) -> str:
        """Edit the most recent or quoted chat images with Qwen-Image-2.1.

        Supports outfit changes, expression/background/pose changes, and
        multi-image fusion (up to 3 images; the first is the edit target).
        Users can also stage reference images first with the mark command and
        refer to them as image 1/2/3 (图一/图二/图三) in the prompt.
        Text-to-image is available via the explicit /qwen 生图 command.

        Args:
            prompt(string): Edit requirement in any language.
        """
        return await self._llm_tool_bridge.edit(event, prompt)
