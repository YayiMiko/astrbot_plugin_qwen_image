# Qwen 绘图（图生图）

这是一个 AstrBot 本地 ComfyUI 图生图插件，使用 Qwen-Image-2.1 参考编辑链路。插件使用独立命令、配置、持久化目录和 LLM 工具名，可以与 Anima / Krea / N5 插件同时启用。

> 文生图正在开发中，暂不可用。`/qwen 生图` 会返回开发中提示。

## 默认工作流

内置 `qwen21_edit` 现在按本机已验证的
`Qwen-Image-2.1-图像编辑-双图参考` 工作流构建 ComfyUI API 图；已保存的
`qwen21_edit` 配置也会使用新图。无需把 ComfyUI 的界面工作流 JSON 粘贴到插件配置里。

- UNET：`qwen_image_2.1_nvfp4.safetensors`
- CLIP：`qwen3vl_8b_nvfp4_heretic.safetensors`，类型 `qwen_image`
- VAE：`qwen_image_2.1_vae_bf16.safetensors`
- 条件编码：`TextEncodeQwenImage21`（聊天入口最多取 3 张图），负面提示词由同一节点编码；模型经过 `QwenImage21Cache`（`auto` / `int8`）
- 默认每张输入图在工作流内最多取 1 MiP 作为参考，小图不放大。**单图和双图/三图**默认都使用编码器输出的目标图 latent，保持第一张图的宽高比，输出最多约 1 MiP。插件配置页可以关闭“限制参考图与默认输出至 1 MP”，改用原图分辨率；高分辨率会增加显存占用。需要固定比例时仍可选择自定义输出尺寸模式。
- 采样：25 steps、CFG 1.0、`euler + simple`
- `SaveImageAdvanced` 输出 PNG，目录前缀：`astrbot/qwen`

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

图片来源优先级：**本次消息自带/引用的图 > 已标记的槽位图**。第一张恒为改图目标，后面为参考（换装/融合/风格参考）。单图及多图默认跟随第一张图宽高比；工作流默认将每张参考图限制到最多 1 MiP，因此默认输出也最多约 1 MiP。关闭配置页的 1 MP 限制后使用原图分辨率，并忽略“输入图最长边”；手动固定输出尺寸设置仍会生效。引用图的 PNG/JPEG 重编码副本会按画面近似去重，避免误判成双图。

## 提示词三层

1. **固定模板**（明确且单一的高频需求，零 LLM 调用）：换装/换表情/换背景/换姿势/改发色/加物件/风格化。遇到否定、复合动作或多图关系时转给视觉 LLM，避免关键词误触发。双图换装以原工作流中的英文提示词为基准，明确图一提供角色、图二提供服装；模板留在代码里。
2. **LLM 改写**（其余一切）：插件 Agent 读取仓库内 `skills/qwen-image-21-prompt-expert/references/edit-policy.md` 作为默认 system prompt。根据输入图和用户语言写清编辑动作、图片分工与必要的保留项，不强制英文或把照片改成二次元。`改写模式` 可选 `auto` / `template_only` / `llm_only`；配置页的自定义 system prompt 会覆盖技能规则，清空可恢复内置。
3. **原样直发**：`原样` / `无优化` 前缀跳过所有改写。

Qwen-Image-2.1 使用自然语言描述而非 tag 串。双图换装默认模板是工作流原有的英文段落；其他固定模板可能保留用户写下的中文细节。供 Codex 等 Agent 使用的完整入口在 `skills/qwen-image-21-prompt-expert/SKILL.md`，按无图文生图、单图编辑、多图参考读取不同规则。插件 Agent 当前仍只执行图生图，并读取该技能的共享编辑规则 `references/edit-policy.md`；技能具备文生图提示词能力不代表 `/qwen 生图` 已实现。

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
