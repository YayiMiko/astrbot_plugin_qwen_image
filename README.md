# Qwen-Image-2.1 绘图插件

这个 AstrBot 插件通过 ComfyUI 使用 Qwen-Image-2.1，支持文生图、单图改图和最多三图参考编辑。AstrBot 只负责接收指令、整理图片、提交工作流和回传结果；提示词增强在 ComfyUI 所在机器上由本地 PE GGUF 完成，不调用 AstrBot 的聊天 LLM，也不默认添加二次元或其它画风。

## 使用

```text
/qwen 生图 雨夜的旧书店
/qwen 改图 把图一角色的衣服换成图二的服装  （同时附图，或先用 /qwen 记图）
/qwen 生图 原样 A red bicycle against a white wall
/qwen 改图 无优化 Only change the sky to sunset
/qwen 记图
/qwen 看图
/qwen 清图
/qwen 状态
```

`原样`、`无优化`、`raw:` 前缀会跳过 PE，直接提交后面的提示词。文生图只响应显式 `/qwen 生图` 等子命令，普通聊天里提到 Qwen 不会自动生图。改图第一张图是目标，后续图片是参考。默认跟随目标图比例并限制参考图至约 1 MP；可在配置页关闭限制或指定输出比例。

## 本地工作流与依赖

- 文生图：`QwenPEGGUF_T2I` → `TextEncodeQwenImage21` → Qwen-Image-2.1 采样 → 保存图片。画幅由插件配置中的宽高比和像素量控制，默认 3:4、约 1 MP，与本机 PE 文生图工作流的固定画幅一致；当前不自动采用 PE 输出的比例建议。
- 改图：上传目标图/参考图 → `QwenPEGGUF_Edit` 读取这些图片和原始要求 → `TextEncodeQwenImage21` → Qwen-Image-2.1 编辑采样 → 保存图片。
- 本机参考工作流：`Qwen-Image-2.1-PE-GGUF-文生图.json`、`Qwen-Image-2.1-PE-GGUF-图像编辑-双图参考.json`。插件提交的是同等关键节点的 API 图，不依赖这些界面 JSON 的绝对路径。
- ComfyUI 必须安装 Qwen-Image-2.1 的编码、缓存和保存节点，以及本机 `ComfyUI-QwenPE-GGUF`。GGUF 默认文件名为 `pe-t2i/pe_t2i_heretic-Q4_K_M.gguf` 和 `pe-i2i/pe_i2i_heretic-Q4_K_M.gguf`，可在插件配置页修改。

PE 自定义节点、llama-server、GGUF 权重和 Qwen-Image-2.1 模型不包含在本插件仓库中；安装前应在 ComfyUI 中准备好这些依赖，并用 `/qwen 状态` 核查节点与模型。

PE 运行在 ComfyUI 机器上。即使 AstrBot 在服务器，配置中的 PE 地址 `http://127.0.0.1:8189` 也是 **ComfyUI 本机** 的服务地址，不是 AstrBot 容器地址。AstrBot 的 `comfyui_base_url` 则仍需填写服务器能访问的 ComfyUI 地址。PE 失败或返回不可解析内容时工作流会报错，不会悄悄把原始中文当增强结果继续生成。PE 与采样共用至少 1800 秒的任务超时预算。

内置改图使用 `qwen21_edit`，默认采样 25 步、CFG 1.0、`euler + simple`，模型文件名在配置页核对。自定义编辑工作流仍按原有节点绑定契约运行；由于其节点结构未知，目前自定义工作流直接使用原始提示词，不插入 PE。要使用本地 PE，请关闭自定义工作流并使用内置链路。

自定义编辑图按节点编号绑定：`LoadImage` 依序接收目标图和参考图；第一个含 `text` 输入的编码节点接收正向提示词，第二个（若有）接收负面提示词。所有 `SaveImage` 前缀改为 `astrbot/qwen_custom`，采样节点的 seed/steps/CFG 按请求覆盖。图像编码器与目标图的连接由工作流作者负责。

`/qwen 诊断` 和 `/qwen 调试状态` 保留。QQ 侧发送成功时只发图片，不再附加“已生成并发送”文字。
