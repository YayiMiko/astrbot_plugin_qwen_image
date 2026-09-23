from __future__ import annotations

import asyncio
import sys
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[1]
AGENT_TOOLS_DIR = PLUGIN_DIR / "agent_tools"
sys.path.insert(0, str(PLUGIN_DIR))
sys.path.insert(0, str(AGENT_TOOLS_DIR))

from agent_tools.comfyui_agent import (  # noqa: E402
    DEFAULT_CONFIG as COMFYUI_AGENT_DEFAULT_CONFIG,
)
from agent_tools.comfyui_agent import (  # noqa: E402
    ROOT as COMFYUI_AGENT_ROOT,
)
from agent_tools.comfyui_agent import _flatten_config as flatten_agent_config  # noqa: E402
from agent_tools.comfyui_sizes import resolve_output_size  # noqa: E402
from agent_tools.comfyui_workflows import (  # noqa: E402
    MAX_EDIT_IMAGES,
    build_edit_workflow,
    custom_qwen_edit_workflow,
    custom_workflow_source,
    describe_custom_workflow,
    qwen21_edit_workflow,
)
from command_router import (  # noqa: E402
    help_text,
    parse_hard_route,
)
from image_slots import (  # noqa: E402
    detect_target_mention,
    find_image_mentions,
    remap_mentions,
    reorder_target_first,
    sanitize_session_key,
    slot_key,
    ImageSlotStore,
)
from prompt_templates import (  # noqa: E402
    TWO_IMAGE_OUTFIT_PROMPT,
    match_template,
    template_names,
)
from prompt_pipeline import (  # noqa: E402
    EDIT_FALLBACK_SKELETON,
    QWEN_REWRITE_SYSTEM_PROMPT,
    PromptPipeline,
    strip_raw_prefix,
)


def test_default_rewriter_loads_bundled_edit_skill() -> None:
    policy = (
        PLUGIN_DIR
        / "skills"
        / "qwen-image-21-prompt-expert"
        / "references"
        / "edit-policy.md"
    )
    assert QWEN_REWRITE_SYSTEM_PROMPT == policy.read_text(encoding="utf-8").strip()


def _config(**overrides):
    base = {
        "unet_name": "qwen_image_2.1_nvfp4.safetensors",
        "clip_name": "qwen3vl_8b_nvfp4_heretic.safetensors",
        "vae_name": "qwen_image_2.1_vae_bf16.safetensors",
        "sampler_name": "euler",
        "scheduler": "simple",
    }
    base.update(overrides)
    return base


def test_edit_workflow_matches_validated_graph() -> None:
    workflow = qwen21_edit_workflow(
        _config(), "Keep everything.", ["target.png"], 25, 1.0, 42
    )

    assert workflow["451"]["class_type"] == "UNETLoader"
    assert workflow["451"]["inputs"]["unet_name"] == (
        "qwen_image_2.1_nvfp4.safetensors"
    )
    assert workflow["453"]["inputs"] == {
        "clip_name": "qwen3vl_8b_nvfp4_heretic.safetensors",
        "type": "qwen_image",
        "device": "default",
    }
    assert workflow["454"]["inputs"] == {
        "vae_name": "qwen_image_2.1_vae_bf16.safetensors"
    }
    assert workflow["469"]["inputs"] == {
        "model": ["451", 0],
        "device": "auto",
        "dtype": "int8",
    }
    assert workflow["470"] == {
        "class_type": "LoadImage",
        "inputs": {"image": "target.png"},
    }
    assert workflow["485"]["inputs"]["expression"] == "min(c, a*b/1048576)"
    assert workflow["477"]["inputs"]["megapixels"] == ["485", 0]
    assert workflow["474"]["class_type"] == "TextEncodeQwenImage21"
    assert workflow["474"]["inputs"]["images.image_1"] == ["477", 0]
    assert workflow["474"]["inputs"]["prompt"] == "Keep everything."
    assert workflow["474"]["inputs"]["vae"] == ["454", 0]
    assert workflow["474"]["inputs"]["negative_prompt"] == ""
    sampler = workflow["458"]["inputs"]
    assert sampler["model"] == ["469", 0]
    assert sampler["positive"] == ["474", 0]
    assert sampler["negative"] == ["474", 1]
    assert sampler["latent_image"] == ["474", 2]
    assert sampler["steps"] == 25
    assert sampler["cfg"] == 1.0
    assert sampler["sampler_name"] == "euler"
    assert sampler["scheduler"] == "simple"
    assert sampler["denoise"] == 1.0
    assert workflow["457"]["inputs"] == {"samples": ["458", 0], "vae": ["454", 0]}
    assert workflow["461"]["class_type"] == "SaveImageAdvanced"
    assert workflow["461"]["inputs"]["filename_prefix"] == "astrbot/qwen"


def test_edit_workflow_wires_up_to_three_reference_images() -> None:
    workflow = qwen21_edit_workflow(
        _config(), "prompt", ["a.png", "b.png", "c.png", "d.png"], 25, 1.0, 7
    )
    assert workflow["474"]["inputs"]["images.image_1"] == ["477", 0]
    assert workflow["474"]["inputs"]["images.image_2"] == ["479", 0]
    assert workflow["474"]["inputs"]["images.image_3"] == ["493", 0]
    assert workflow["475"]["inputs"] == {"image": "b.png"}
    assert workflow["490"]["inputs"] == {"image": "c.png"}
    assert "d.png" not in str(workflow)
    assert MAX_EDIT_IMAGES == 3


def test_cli_agent_defaults_match_validated_models() -> None:
    assert COMFYUI_AGENT_ROOT == PLUGIN_DIR.parents[2]
    assert (
        COMFYUI_AGENT_DEFAULT_CONFIG["unet_name"] == "qwen_image_2.1_nvfp4.safetensors"
    )
    assert (
        COMFYUI_AGENT_DEFAULT_CONFIG["clip_name"]
        == "qwen3vl_8b_nvfp4_heretic.safetensors"
    )
    assert (
        COMFYUI_AGENT_DEFAULT_CONFIG["vae_name"]
        == "qwen_image_2.1_vae_bf16.safetensors"
    )
    assert COMFYUI_AGENT_DEFAULT_CONFIG["steps"] == 25
    assert COMFYUI_AGENT_DEFAULT_CONFIG["cfg"] == 1.0
    assert COMFYUI_AGENT_DEFAULT_CONFIG["workflow"] == "qwen21_edit"
    assert COMFYUI_AGENT_DEFAULT_CONFIG["output_size_mode"] == "aspect"
    assert COMFYUI_AGENT_DEFAULT_CONFIG["output_aspect"] == "4:3"
    assert COMFYUI_AGENT_DEFAULT_CONFIG["output_megapixels"] == 1.0
    assert COMFYUI_AGENT_DEFAULT_CONFIG["single_image_size_mode"] == "target"


def test_cli_agent_flattens_qwen_grouped_config() -> None:
    flat = flatten_agent_config(
        {"qwen_models": {"unet_name": "custom.safetensors"}, "timeout": 600}
    )
    assert flat["unet_name"] == "custom.safetensors"
    assert flat["timeout"] == 600
    assert "qwen_models" not in flat


def test_hard_route_edit_and_t2i_stub() -> None:
    assert parse_hard_route("/qwen 改图 让图一换上红裙") == (
        "edit",
        "让图一换上红裙",
    )
    assert parse_hard_route("/qwen 编辑 为图中角色穿上吸血鬼贵族礼服") == (
        "edit",
        "为图中角色穿上吸血鬼贵族礼服",
    )
    assert parse_hard_route("/qwen 换装 图二的衣服") == ("edit", "图二的衣服")
    assert parse_hard_route("/qwen 生图 一只猫") == ("t2i_stub", "一只猫")
    assert parse_hard_route("/qwen 状态") == ("status", "")
    assert parse_hard_route("/qwen") == ("help", "")
    # Other plugins' prefixes must not be captured.
    assert parse_hard_route("/krea 生图 一只猫") is None
    assert parse_hard_route("/anm 状态") is None
    assert parse_hard_route("/n5 一只猫") is None


def test_natural_route_edit() -> None:
    assert parse_hard_route("用qwen改图换个背景") == ("edit", "换个背景")


def test_slot_command_routes() -> None:
    assert parse_hard_route("/qwen 记图") == ("mark_slots", "")
    assert parse_hard_route("/qwen 标记") == ("mark_slots", "")
    assert parse_hard_route("/qwen 看图") == ("show_slots", "")
    assert parse_hard_route("/qwen 清图") == ("clear_slots", "")
    assert parse_hard_route("用qwen记图") == ("mark_slots", "")
    text = help_text(True)
    assert "/qwen 记图" in text
    assert "/qwen 看图" in text
    assert "/qwen 清图" in text
    # 状态类只留状态在 help 显示；诊断/调试保留功能但折叠。
    assert "/qwen 状态" in text
    assert "诊断" not in text
    assert "调试" not in text
    # 改图与编辑是同一功能，help 只展示一个入口。
    assert "改图" in text and "编辑" in text


def test_help_mentions_edit_and_t2i_unavailable() -> None:
    text = help_text(True)
    assert "/qwen 改图" in text
    assert "文生图正在开发中" in text or "开发中" in text


def test_strip_raw_prefix() -> None:
    matched, rest = strip_raw_prefix("无优化 my prompt")
    assert matched and rest == "my prompt"
    matched, rest = strip_raw_prefix("原样：keep this")
    assert matched and rest == "keep this"
    matched, rest = strip_raw_prefix("让图一换衣服")
    assert not matched and rest == "让图一换衣服"


def _pipeline(**overrides):
    config = {"prompt_optimize_enabled": True}
    context = overrides.pop("context", None)
    config.update(overrides)

    def _bool(key, default=False):
        return bool(config.get(key, default))

    def _int(key, default=0):
        try:
            return int(config.get(key, default))
        except (TypeError, ValueError):
            return default

    def _str(key, default=""):
        value = config.get(key, default)
        return str(value if value is not None else default)

    class _Logger:
        def info(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

    return PromptPipeline(
        context=context,
        config=config,
        logger=_Logger(),
        get_bool=_bool,
        get_int=_int,
        get_str=_str,
        shorten=lambda text, limit=1800: str(text or "")[:limit],
    )


def test_fallback_skeleton_preserves_target_without_llm() -> None:
    pipeline = _pipeline(prompt_optimize_enabled=False)
    result = asyncio.run(pipeline.build(None, "换上红裙", mode="img2img"))
    assert result.final_prompt == EDIT_FALLBACK_SKELETON.format(change="换上红裙")
    assert "supplied target image" in result.final_prompt
    assert "Preserve its identity" in result.final_prompt
    assert "1girl" not in result.final_prompt
    assert result.summary["llm_used"] is False


def test_raw_mode_passes_prompt_through() -> None:
    pipeline = _pipeline()
    result = asyncio.run(pipeline.build(None, "无优化 custom english prompt"))
    assert result.final_prompt == "custom english prompt"
    assert result.summary.get("raw_mode") is True


def test_template_tier_hit_without_llm() -> None:
    pipeline = _pipeline()
    result = asyncio.run(
        pipeline.build(None, "把背景换成黄昏海滩", mode="img2img", image_count=1)
    )
    assert result.summary.get("tier") == "template:background"
    assert "背景替换为黄昏海滩" in result.final_prompt
    assert "保留前景主体" in result.final_prompt
    assert "黄昏海滩" in result.final_prompt
    assert "1girl" not in result.final_prompt
    assert result.summary.get("llm_used") is False


def test_template_outfit_uses_ref_default() -> None:
    pipeline = _pipeline()
    result = asyncio.run(pipeline.build(None, "换装", mode="img2img", image_count=2))
    assert result.summary.get("tier") == "template:outfit"
    assert "<image2>" in result.final_prompt


def test_template_only_miss_falls_back_without_llm() -> None:
    pipeline = _pipeline(prompt_mode="template_only")
    result = asyncio.run(
        pipeline.build(None, "让画面更有电影感", mode="img2img", image_count=1)
    )
    assert result.summary.get("tier") == "fallback"
    assert result.summary.get("skipped_reason") == "template_miss"
    assert result.summary.get("llm_used") is False


def test_llm_only_skips_templates() -> None:
    pipeline = _pipeline(prompt_mode="llm_only")
    result = asyncio.run(
        pipeline.build(None, "把背景换成黄昏海滩", mode="img2img", image_count=1)
    )
    assert result.summary.get("tier") == "fallback"
    assert result.summary.get("skipped_reason") == "rewrite_failed"


def test_production_case_reordered_prompt_builds_correct_roles() -> None:
    # Production bug: "为图2角色穿上图1的衣服" with supply [图1, 图2].
    # After reorder_target_first the prompt becomes target-first and the
    # template must keep <image1> as target with outfit from <image2>.
    paths, numbers, prompt = reorder_target_first(
        ["outfit.png", "character.png"],
        [1, 2],
        "为图2角色穿上图1的衣服",
    )
    assert paths == ["character.png", "outfit.png"]
    pipeline = _pipeline()
    result = asyncio.run(pipeline.build(None, prompt, mode="img2img", image_count=2))
    assert result.summary.get("tier") == "template:outfit"
    assert result.final_prompt == TWO_IMAGE_OUTFIT_PROMPT


def test_outfit_parenthetical_denoised() -> None:
    matched = match_template("为图2角色穿上图1的衣服", 2)
    assert matched is not None
    assert "(图" not in matched[1]
    assert "clothing from <image2>" in matched[1]
    assert matched[1] == TWO_IMAGE_OUTFIT_PROMPT
    assert "faithfully" in matched[1]
    matched = match_template("穿上红色礼服", 2)
    assert matched is not None
    assert "红色礼服" in matched[1]
    assert "(图" not in matched[1]
    single = match_template("穿上红色礼服", 1)
    assert single is not None
    assert "<image2>" not in single[1]


def test_vision_outfit_routing_passes_images() -> None:
    import types

    calls = {}

    class _VisionContext:
        async def llm_generate(self, **kwargs):
            calls.update(kwargs)
            return types.SimpleNamespace(
                completion_text=(
                    "Keep the character and pose in <image1> unchanged, "
                    "dress the character in the white military jacket with "
                    "gold trim and blue shoulder cape from <image2>."
                )
            )

    pipeline = _pipeline(context=_VisionContext())
    result = asyncio.run(
        pipeline.build(
            None,
            "为图2角色穿上图1的衣服",
            mode="img2img",
            image_count=2,
            image_paths=["target.png", "ref.png"],
        )
    )
    assert result.summary.get("tier") == "llm:vision"
    assert calls["image_urls"] == ["target.png", "ref.png"]
    assert "distinctive garments and details you can actually see" in calls["prompt"]
    assert TWO_IMAGE_OUTFIT_PROMPT in calls["prompt"]
    assert calls["system_prompt"] == QWEN_REWRITE_SYSTEM_PROMPT
    assert "white military jacket" in result.final_prompt


def test_vision_failure_falls_back_to_template() -> None:
    class _BrokenContext:
        async def llm_generate(self, **kwargs):
            raise RuntimeError("provider down")

    pipeline = _pipeline(context=_BrokenContext())
    result = asyncio.run(
        pipeline.build(
            None,
            "为图2角色穿上图1的衣服",
            mode="img2img",
            image_count=2,
            image_paths=["target.png", "ref.png"],
        )
    )
    assert result.summary.get("tier") == "template:outfit"
    assert result.summary.get("skipped_reason") == "rewrite_failed"
    assert "clothing from <image2>" in result.final_prompt


def test_template_table_covers_skill_types() -> None:
    names = template_names()
    for expected in (
        "outfit",
        "expression",
        "background",
        "pose",
        "hair",
        "prop",
        "style",
    ):
        assert expected in names
    matched = match_template("改成微笑表情", 1)
    assert matched is not None and matched[0] == "expression"
    matched = match_template("发色改成银色", 1)
    assert matched is not None and matched[0] == "hair"
    assert match_template("随便聊聊今晚吃啥", 1) is None


def test_templates_defer_negated_compound_and_multireference_edits() -> None:
    assert match_template("不要改背景", 1) is None
    assert match_template("换背景并穿上红裙", 1) is None
    assert match_template("把背景换成海滩，保留人物", 1) is None
    assert match_template("把背景换成海滩", 2) is None
    assert match_template("更加强烈的光线", 1) is None
    assert match_template("换装", 1) is None
    assert match_template("换衣服", 1) is None
    assert match_template("背景加入一棵树", 1) is None
    assert match_template("change the background to a beach", 1) is None


def test_compound_edit_reaches_vision_rewriter() -> None:
    import types

    calls = {}

    class _VisionContext:
        async def llm_generate(self, **kwargs):
            calls.update(kwargs)
            return types.SimpleNamespace(
                completion_text="同时更换背景和服装，保留人物身份。"
            )

    pipeline = _pipeline(context=_VisionContext())
    result = asyncio.run(
        pipeline.build(
            None,
            "把背景换成海滩，同时穿上红裙",
            image_count=1,
            image_paths=["target.png"],
        )
    )
    assert result.summary["tier"] == "llm"
    assert calls["image_urls"] == ["target.png"]
    assert result.final_prompt == "同时更换背景和服装，保留人物身份。"


def test_single_image_templates_keep_user_language_and_specific_change() -> None:
    assert "表情改为微笑" in match_template("改成微笑表情", 1)[1]
    assert "发色改为银色" in match_template("发色改成银色", 1)[1]
    assert "人物坐下" in match_template("让她坐下", 1)[1]
    assert "添加一顶帽子" in match_template("给她加一顶帽子", 1)[1]
    assert "水彩风格" in match_template("画成水彩风格", 1)[1]


def test_find_image_mentions() -> None:
    assert find_image_mentions("为图1角色穿上图二的衣服") == [1, 2]
    assert find_image_mentions("把第一张的背景换掉，参考第三张") == [1, 3]
    assert find_image_mentions("edit image 2 outfit") == [2]
    assert find_image_mentions("随便改改") == []
    assert sanitize_session_key("a/b?c:d") == "a_b_c_d"


def test_slot_key_is_per_user() -> None:
    assert slot_key("group1", "userA") != slot_key("group1", "userB")
    assert slot_key("group1", "userA") == slot_key("group1", "userA")
    assert slot_key("group1", "userA") != slot_key("group2", "userA")
    assert "__" in slot_key("group1", "userA")


def test_slots_isolated_per_user(tmp_path) -> None:
    store = _slot_store(tmp_path)
    alice = _touch(tmp_path / "alice.png")
    bob = _touch(tmp_path / "bob.png")
    store.save(slot_key("group1", "alice"), [alice])
    store.save(slot_key("group1", "bob"), [bob])
    assert [s["path"] for s in store.read(slot_key("group1", "alice"))] == [alice]
    assert [s["path"] for s in store.read(slot_key("group1", "bob"))] == [bob]


def test_detect_target_mention() -> None:
    assert detect_target_mention("为图2角色穿上图1的衣服") == 2
    assert detect_target_mention("为图一角色换个背景") == 1
    assert detect_target_mention("把图2背景换成海滩") == 2
    assert detect_target_mention("让图一的角色穿上图二的衣服") is None
    assert detect_target_mention("随便改改") is None


def test_reorder_target_first_swaps_roles() -> None:
    paths, numbers, prompt = reorder_target_first(
        ["slot1.png", "slot2.png"],
        [1, 2],
        "为图2角色穿上图1的衣服",
    )
    assert paths == ["slot2.png", "slot1.png"]
    assert numbers == [2, 1]
    assert prompt == "为图1角色穿上图2的衣服"


def test_reorder_target_first_noop_cases() -> None:
    assert reorder_target_first(["a.png"], [1], "为图1改改") == (
        ["a.png"],
        [1],
        "为图1改改",
    )
    assert reorder_target_first(
        ["a.png", "b.png"], [1, 2], "让图一的角色穿上图二的衣服"
    ) == (["a.png", "b.png"], [1, 2], "让图一的角色穿上图二的衣服")
    assert reorder_target_first(["a.png", "b.png"], [1, 2], "为图3改改") == (
        ["a.png", "b.png"],
        [1, 2],
        "为图3改改",
    )


def test_remap_mentions_preserves_forms() -> None:
    assert remap_mentions("第一张不动，参考图一", {1: 2}) == "第二张不动，参考图二"
    assert remap_mentions("edit image 1 first", {1: 2, 2: 1}) == "edit image 2 first"


def _slot_store(tmp_path, **overrides):
    class _Logger:
        def info(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

    kwargs = {"ttl_minutes": 30, "max_slots": 3}
    kwargs.update(overrides)
    return ImageSlotStore(tmp_path / "slots", _Logger(), **kwargs)


def _touch(path) -> str:
    path.write_bytes(b"fake-image")
    return str(path)


def test_slot_save_read_clear(tmp_path) -> None:
    store = _slot_store(tmp_path)
    first = _touch(tmp_path / "a.png")
    second = _touch(tmp_path / "b.png")
    result = store.save("sess1", [first, second])
    assert result["evicted"] == 0
    slots = store.read("sess1")
    assert [slot["path"] for slot in slots] == [first, second]
    assert store.clear("sess1") == 2
    assert store.read("sess1") == []


def test_slot_fifo_eviction_and_dedup(tmp_path) -> None:
    store = _slot_store(tmp_path)
    paths = [_touch(tmp_path / f"{name}.png") for name in ("a", "b", "c", "d")]
    result = store.save("sess1", paths)
    assert result["evicted"] == 1
    assert [slot["path"] for slot in store.read("sess1")] == paths[1:]
    store.save("sess1", [paths[1]])
    assert len(store.read("sess1")) == 3


def test_slot_expiry(tmp_path) -> None:
    import json
    import time

    store = _slot_store(tmp_path, ttl_minutes=30)
    path = _touch(tmp_path / "a.png")
    store.save("sess1", [path])
    slot_file = tmp_path / "slots" / "sess1.json"
    data = json.loads(slot_file.read_text(encoding="utf-8"))
    data[0]["saved_at"] = time.time() - 31 * 60
    slot_file.write_text(json.dumps(data), encoding="utf-8")
    assert store.read("sess1") == []


def _custom_graph():
    return {
        "1": {
            "class_type": "LoadImage",
            "inputs": {"image": "old_target.png"},
        },
        "2": {
            "class_type": "LoadImage",
            "inputs": {"image": "old_ref.png"},
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "old positive", "clip": ["0", 0]},
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "old negative", "clip": ["0", 0]},
        },
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "seed": 1,
                "steps": 99,
                "cfg": 9.0,
                "noise_seed": 2,
            },
        },
        "6": {
            "class_type": "SaveImage",
            "inputs": {"images": ["5", 0], "filename_prefix": "ComfyUI"},
        },
    }


def test_custom_workflow_binding() -> None:
    import json

    config = {
        "custom_workflow_enabled": True,
        "custom_workflow_path": "",
        "custom_workflow_json": json.dumps(_custom_graph()),
    }
    graph = custom_qwen_edit_workflow(
        config, "new prompt", ["t.png", "r.png"], 8, 1.0, 42, "neg"
    )
    assert graph["1"]["inputs"]["image"] == "t.png"
    assert graph["2"]["inputs"]["image"] == "r.png"
    assert graph["3"]["inputs"]["text"] == "new prompt"
    assert graph["4"]["inputs"]["text"] == "neg"
    assert graph["5"]["inputs"]["seed"] == 42
    assert graph["5"]["inputs"]["noise_seed"] == 42
    assert graph["5"]["inputs"]["steps"] == 8
    assert graph["5"]["inputs"]["cfg"] == 1.0
    assert graph["6"]["inputs"]["filename_prefix"] == "astrbot/qwen_custom"


def test_custom_workflow_binding_mismatch() -> None:
    import json

    import pytest

    config = {
        "custom_workflow_enabled": True,
        "custom_workflow_path": "",
        "custom_workflow_json": json.dumps(_custom_graph()),
    }
    with pytest.raises(ValueError, match="custom_workflow_image_binding_mismatch"):
        custom_qwen_edit_workflow(config, "p", ["a.png", "b.png", "c.png"], 8, 1.0, 1)
    with pytest.raises(ValueError, match="custom_workflow_no_save_node"):
        graph = _custom_graph()
        del graph["6"]
        bad = dict(config, custom_workflow_json=json.dumps(graph))
        custom_qwen_edit_workflow(bad, "p", ["a.png"], 8, 1.0, 1)


def test_custom_workflow_source_precedence(tmp_path) -> None:
    import json

    path_file = tmp_path / "graph.json"
    path_file.write_text(json.dumps(_custom_graph()), encoding="utf-8")
    config = {
        "custom_workflow_path": str(path_file),
        "custom_workflow_json": json.dumps({"9": {"class_type": "X"}}),
    }
    source, body = custom_workflow_source(config)
    assert source.startswith("path:")
    assert "1" in body
    config = {"custom_workflow_path": "", "custom_workflow_json": ""}
    import pytest

    with pytest.raises(ValueError, match="custom_workflow_not_configured"):
        custom_workflow_source(config)


def test_describe_custom_workflow() -> None:
    import json

    config = {
        "custom_workflow_path": "",
        "custom_workflow_json": json.dumps(_custom_graph()),
    }
    summary = describe_custom_workflow(config)
    assert summary["ok"] is True
    assert summary["load_slots"] == 2
    assert summary["save_nodes"] == 1
    assert summary["positive_node"] == "3"
    bad = describe_custom_workflow(
        {"custom_workflow_path": "", "custom_workflow_json": "{}"}
    )
    assert bad["ok"] is False


def test_build_edit_workflow_dispatch() -> None:
    import pytest

    builtin = build_edit_workflow(_config(), "p", ["a.png"], 25, 1.0, 1)
    assert builtin["461"]["inputs"]["filename_prefix"] == "astrbot/qwen"
    assert "456" not in builtin
    with pytest.raises(ValueError, match="unsupported_workflow"):
        build_edit_workflow(
            dict(_config(), workflow="qwen21_t2i"), "p", ["a.png"], 25, 1.0, 1
        )


def test_resolve_output_size() -> None:
    assert resolve_output_size({}) is None
    assert resolve_output_size({"output_size_mode": "target"}) is None
    assert (
        resolve_output_size(
            {
                "output_size_mode": "aspect",
                "output_aspect": "4:3",
                "output_megapixels": 1.0,
            },
            image_count=1,
        )
        is None
    )
    assert resolve_output_size(
        {
            "output_size_mode": "target",
            "single_image_size_mode": "configured",
            "output_aspect": "4:3",
            "output_megapixels": 1.0,
        },
        image_count=1,
    ) == (1152, 864)
    assert resolve_output_size(
        {
            "output_size_mode": "aspect",
            "output_aspect": "3:4",
            "output_megapixels": 1.0,
        }
    ) == (864, 1152)
    assert resolve_output_size(
        {"output_size_mode": "aspect", "output_aspect": "1:1", "output_megapixels": 1.0}
    ) == (992, 992)
    assert (
        resolve_output_size(
            {
                "output_size_mode": "aspect",
                "output_aspect": "9:99",
                "output_megapixels": 1.0,
            }
        )
        is None
    )
    assert (
        resolve_output_size(
            {
                "output_size_mode": "aspect",
                "output_aspect": "3:4",
                "output_megapixels": 0,
            }
        )
        is None
    )


def test_edit_workflow_forced_output_size() -> None:
    workflow = qwen21_edit_workflow(
        _config(), "p", ["t.png"], 25, 1.0, 1, output_size=(864, 1152)
    )
    assert workflow["456"] == {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": 864, "height": 1152, "batch_size": 1},
    }
    assert workflow["458"]["inputs"]["latent_image"] == ["456", 0]
    assert workflow["474"]["inputs"]["images.image_1"] == ["477", 0]


def test_check_commands_removed_from_chat() -> None:
    # 验工作流/取日志已移出 QQ 侧（改配置页实现），不再参与路由；
    # 原文本按裸 /qwen 文本兜底为 edit。
    assert parse_hard_route("/qwen 验工作流") == ("edit", "验工作流")
    assert parse_hard_route("/qwen 取日志") == ("edit", "取日志")
    assert parse_hard_route("用qwen验工作流") is None
    text = help_text(True)
    assert "/qwen 验工作流" not in text
    assert "/qwen 取日志" not in text
