from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from astrbot.core.utils.astrbot_path import get_astrbot_root

try:
    from .agent_tools.comfyui_workflows import MAX_EDIT_IMAGES
    from .comfyui_runtime import ComfyUIRuntime
    from .command_actions import CommandActionHandler
    from .image_inputs import ImageInputResolver
    from .image_slots import ImageSlotStore
    from .llm_tool_bridge import LLMToolBridge
    from .task_state import TaskRecorder
except Exception:  # pragma: no cover - fallback for direct script-style imports.
    from agent_tools.comfyui_workflows import MAX_EDIT_IMAGES
    from comfyui_runtime import ComfyUIRuntime
    from command_actions import CommandActionHandler
    from image_inputs import ImageInputResolver
    from image_slots import ImageSlotStore
    from llm_tool_bridge import LLMToolBridge
    from task_state import TaskRecorder


@dataclass(frozen=True)
class ServicePaths:
    """Resolved filesystem paths used by Qwen services.

    Args:
        root: AstrBot root directory.
        plugin_dir: Current plugin directory.
        tool: Main ComfyUI helper script path.
        python: Preferred Python interpreter path.
        workspace: AstrBot workspace directory.
        inputs: Image input storage directory.
        plugin_data: Plugin persistent data directory.
    """

    root: Path
    plugin_dir: Path
    tool: Path
    python: Path
    workspace: Path
    inputs: Path
    plugin_data: Path


@dataclass(frozen=True)
class QwenServices:
    """Constructed services used by the plugin entry point.

    Args:
        paths: Resolved filesystem paths.
        runtime: Local ComfyUI process/tool runtime.
        task_recorder: Latest task summary recorder.
        image_inputs: Image input resolver.
        action_handler: Chat command action handler.
        llm_tool_bridge: LLM tool bridge.
    """

    paths: ServicePaths
    runtime: ComfyUIRuntime
    task_recorder: TaskRecorder
    image_inputs: ImageInputResolver
    slot_store: ImageSlotStore
    action_handler: CommandActionHandler
    llm_tool_bridge: LLMToolBridge


def resolve_service_paths() -> ServicePaths:
    """Resolve Qwen plugin runtime paths.

    Returns:
        Resolved path bundle for plugin services.
    """
    root = Path(get_astrbot_root())
    plugin_dir = Path(__file__).resolve().parent
    tool = plugin_dir / "agent_tools" / "comfyui_agent.py"
    if not tool.exists():
        tool = root / "agent_tools" / "comfyui_agent.py"
    workspace = root / "workspace"
    return ServicePaths(
        root=root,
        plugin_dir=plugin_dir,
        tool=tool,
        python=root / ".venv" / "Scripts" / "python.exe",
        workspace=workspace,
        inputs=workspace / "inputs",
        plugin_data=root / "data" / "plugin_data" / "astrbot_plugin_qwen_image",
    )


def build_services(
    *,
    context: Any,
    config: dict[str, Any],
    config_store: Any = None,
    logger: Any,
    get_bool: Callable[[str, bool], bool],
    get_int: Callable[[str, int], int],
    get_str: Callable[[str, str], str],
    shorten: Callable[[str, int], str],
    is_allowed: Callable[[Any], bool],
    edit: Callable[[Any, str], Any],
) -> QwenServices:
    """Build all Qwen services for the plugin entry point.

    Args:
        context: AstrBot plugin context.
        config: Plugin configuration dict.
        config_store: Original mutable plugin config object, when available.
        logger: Logger compatible with AstrBot logger methods.
        get_bool: Config boolean accessor.
        get_int: Config integer accessor.
        get_str: Config string accessor.
        shorten: Text-shortening helper.
        is_allowed: Permission checker.
        edit: Edit callback for LLM tools.

    Returns:
        Constructed service container.
    """
    paths = resolve_service_paths()
    runtime = ComfyUIRuntime(
        root=paths.root,
        tool=paths.tool,
        prompt_tool=paths.plugin_dir / "agent_tools" / "comfyui_agent.py",
        python=paths.python,
        config=config,
        logger=logger,
        get_bool=get_bool,
        get_int=get_int,
        get_str=get_str,
    )
    task_recorder = TaskRecorder(paths.plugin_data / "last_task.json", logger)

    def _prepare_probe_images() -> list[str]:
        """Copy shipped probe images into workspace inputs for validation."""
        staged: list[str] = []
        probe_dir = paths.plugin_dir / "probe"
        for name in ("probe_target.png", "probe_ref.png"):
            src = probe_dir / name
            if not src.is_file():
                continue
            dst = paths.inputs / f"qwen_probe_{name}"
            try:
                paths.inputs.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
            except OSError as exc:
                logger.warning("[qwen] failed to stage probe image %s: %s", src, exc)
                continue
            staged.append(str(dst))
        return staged

    image_inputs = ImageInputResolver(
        workspace=paths.workspace,
        inputs_dir=paths.inputs,
        logger=logger,
        shorten=shorten,
    )
    slot_store = ImageSlotStore(
        paths.plugin_data / "slots",
        logger,
        ttl_minutes=get_int("slot_ttl_minutes", 30),
        max_slots=MAX_EDIT_IMAGES,
    )
    action_handler = CommandActionHandler(
        config=config,
        config_store=config_store,
        task_recorder=task_recorder,
        is_allowed=is_allowed,
        run_tool=runtime.run_tool,
        ensure_ready=runtime.ensure_ready,
        send_payload=runtime.send_payload,
        event_image_inputs=image_inputs.event_image_inputs,
        image_input_summary=lambda: dict(image_inputs.last_summary),
        slot_store=slot_store,
        prepare_probe_images=_prepare_probe_images,
        get_bool=get_bool,
        shorten=shorten,
    )
    llm_tool_bridge = LLMToolBridge(
        run_tool=runtime.run_tool,
        edit=edit,
    )
    return QwenServices(
        paths=paths,
        runtime=runtime,
        task_recorder=task_recorder,
        image_inputs=image_inputs,
        slot_store=slot_store,
        action_handler=action_handler,
        llm_tool_bridge=llm_tool_bridge,
    )
