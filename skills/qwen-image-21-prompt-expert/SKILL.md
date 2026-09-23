---
name: qwen-image-21-prompt-expert
description: Write or refine Qwen-Image 2.1 image-edit prompts from one or more reference images, including clothing transfer, local edits, style transfer, and reference-based scenes. Text-to-image prompting is not covered yet.
---

# Qwen-Image 2.1 image editing

Use this skill for image-to-image prompting only. Do not apply its edit-preservation rules to text-to-image work. Inspect the images when available; if they are unavailable, keep references to the source images and do not invent their appearance.

Read [the shared edit policy](references/edit-policy.md) before writing or rewriting an edit prompt. The AstrBot Qwen plugin also loads this exact policy as its default LLM system prompt, so changes to it affect both interactive agents and the plugin after deployment. A user-supplied custom plugin system prompt still overrides the default.

## Workflow fit

- Follow the actual image-slot contract. In the AstrBot plugin, `<image1>` is the canvas/target and `<image2>`/`<image3>` are references; other Qwen-Image 2.1 workflows may offer more slots or different canvas placement.
- For an unchanged-picture edit, preserve the input's medium and unedited content. Anime-first is a default only when creating a new scene without a style source; do not turn a photograph into anime by accident.
- Return a copy-ready instruction, not Danbooru tags. Answer in the user's language unless they ask otherwise. Keep parameter advice outside the prompt, and only when useful.
- Image editing is semantic generation, not a pixel-perfect layer operation. If the user needs exact boundaries or identity, suggest an appropriate mask, control, or verification step rather than promising a prompt can guarantee it.

For the plugin's two-image outfit workflow, this tested minimal backbone is a useful starting point, not a mandatory wrapper for unrelated edits:

```text
Put the clothing from <image2> onto the character in <image1>, replacing the outfit they are currently wearing. Keep the character's face, hairstyle, body shape and pose exactly as they appear in <image1>. Reproduce the clothing from <image2> faithfully: the same garment, the same colours, the same pattern, the same details. The clothing should fit the character's body naturally, with correct proportions, believable fabric drape and natural folds at the shoulders, elbows and waist. Keep the original background, camera angle, lighting and art style of <image1> unchanged.
```

This is an English example from the user's working ComfyUI workflow; a Chinese request may be rewritten in Chinese. Do not copy its preserve clauses when the user deliberately changes pose, background, or style.

## Evidence

- [Qwen-Image-2.1 official edit prompt enhancer](https://github.com/QwenLM/Qwen-Image-2.1/blob/main/prompt_rewrite/prompts/system_prompt_edit.txt): operation-first edits, image-grounded details, explicit multi-image markers and roles, preservation without over-description, and language handling.
- [Qwen-Image-2.1 official repository](https://github.com/QwenLM/Qwen-Image-2.1): multi-reference and local-edit capabilities.
- [ComfyUI official image-edit template](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_qwen_image_2_1_image_edit.json): two-image clothing example, image-slot and output-size behavior, CFG 1 default.
- [Recent ComfyUI community edit examples](https://www.reddit.com/r/StableDiffusion/comments/1wlt9t9/qwen_image_21_edit_tips/): concise pose, texture, lighting, and style-transfer requests explicitly name the source and target images. Treat these as anecdotal tests, not model guarantees.

Community examples are useful hypotheses, not guarantees; prefer the current workflow's actual slot contract and output over a prompt recipe copied from another model version.
