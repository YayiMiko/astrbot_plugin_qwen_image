# Qwen 绘图（图生图）

这是一个 AstrBot 本地 ComfyUI 图生图插件，使用 Qwen-Image-2.1 参考编辑链路。插件使用独立命令、配置、持久化目录和 LLM 工具名，可以与 Anima / Krea / N5 插件同时启用。

> 文生图正在开发中，暂不可用。`/qwen 生图` 会返回开发中提示。

## 默认工作流

- UNET：`qwen_image_2.1_nvfp4.safetensors`
- CLIP：`qwen3vl_8b_nvfp4_heretic.safetensors`，类型 `qwen_image`
- VAE：`qwen_image_2.1_vae_bf16.safetensors`
- 条件编码：`TextEncodeQwenImageEditPlus`（最多 3 张参考图）+ 空文本 `CLIPTextEncode` 做负面
- 目标图经 `VAEEncode` 进 `KSampler`，`denoise=1.0`（保持靠 conditioning，不靠 denoise）
- 采样：25 steps、CFG 1.0、`euler + simple`
- 输出目录前缀：`astrbot/qwen`

## 使用

```text
/qwen 改图 让图一的角色穿上图二的衣服    （附两张图，或先记图；改图和编辑是同一个功能）
/qwen 编辑 为图中角色穿上吸血鬼贵族礼服  （附一张图，或引用图片）
/qwen 记图     （把本条带的图/引用的图记为参考图1/2/3，最多 3 张，30 分钟有效；只回显本次新增的图）
/qwen 看图     （查看已标记的参考图顺序)
/qwen 清图     （清空已标记的参考图）
/qwen 状态
/qwen 生图 一只猫   （返回"开发中"提示）
```

`/qwen 诊断` / `/qwen 调试状态` 保留可用，但在指令表里折叠不显示。

图片来源优先级：**本次消息自带/引用的图 > 已标记的槽位图**。第一张恒为改图目标，后面为参考（换装/融合/风格参考）。输出尺寸跟随目标图；超 1024 最长边的输入会在上传前缩小（配置可改）。

## 提示词三层

1. **固定模板**（高频需求，零 LLM 调用）：换装/换表情/换背景/换姿势/改发色/加物件/风格化，直接套保持骨架。模板措辞与编辑节点行为耦合，留在代码里不开放编辑。
2. **LLM 改写**（其余一切）：把中文需求改写成一段英文画面描述。`改写模式` 可选 `auto` / `template_only` / `llm_only`；`改写 system prompt` 可在配置页改，清空恢复内置。
3. **原样直发**：`原样` / `无优化` 前缀跳过所有改写。

Qwen-Image-2.1 使用自然语言描述而非 tag 串（`1girl, solo` 这类写法会让它变差），三层输出都是英文段落。

## 自定义工作流（给有 ComfyUI 经验的用户）

内置 `qwen21_edit` 开箱即用；也可在配置页开启自定义并填写 JSON 路径（相对插件目录，服务器上是服务器路径）或粘贴内联 JSON。**绑定契约**：

- `LoadImage` 节点按编号顺序接收上传图：第一个=编辑目标，后面=参考；节点数不得少于实际图片数。
- 第一个含 `text` 的 `*TextEncode*` 节点收正向提示词，第二个（如果有）收负面。
- 所有 `SaveImage` 前缀会被改写为 `astrbot/qwen_custom`；所有 `seed` / `noise_seed` 会被重随机。
- `VAEEncode` 的目标接线是作者的责任；没连到第一个 LoadImage 时只记警告。

探针校验（8 步快跑）与失败日志导出以后会做到插件配置页；目前 `check_workflow` / `fetch_check_log` 作为内部接口保留，QQ 侧不暴露。
## 部署说明

AstrBot 与 ComfyUI 同机时，默认地址为 `http://127.0.0.1:8188`。AstrBot 位于服务器、ComfyUI 位于本地 Windows 时，应填写 AstrBot 能访问到的局域网或 Tailscale 地址（ComfyUI 需 `--listen 0.0.0.0` 启动，否则服务器报连接超时）；容器内的启动命令不能直接启动远端 Windows ComfyUI。

## 路线图

文生图（`qwen21_t2i`）以后会以新的 workflow 名接入同一套分发（未知 workflow 名现在会明确报错而不是静默跑错图）。
