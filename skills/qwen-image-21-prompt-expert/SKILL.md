---
name: qwen-image-21-prompt-expert
description: Write Qwen-Image 2.1 prompts for text-to-image, single-image editing, and multi-image reference editing in ComfyUI or an agent workflow.
---

# Qwen-Image 2.1 prompts

Use natural-language descriptions, not Danbooru-style tag strings. Select the mode from the images actually supplied to the destination workflow, not merely from image names in the user's text:

- **No input image — text-to-image:** Read [the text-to-image policy](references/t2i-policy.md). Expand a short theme into a complete, coherent finished image. In this user's workspace, default to an anime illustration unless the request calls for another medium.
- **One input image — single-image editing:** Read [the edit policy](references/edit-policy.md). Inspect the image when available. Name the change and preserve only the relevant unaffected content and the source medium.
- **Two or more input images — multi-reference editing:** Read [the edit policy](references/edit-policy.md) and [the multi-reference guide](references/multi-reference.md). Make the role of each used image explicit; distinguish editing an existing canvas from composing a new scene from references.

If an image cannot be inspected, do not invent its appearance; refer to it by its image slot and ask for missing role information only when the task cannot be interpreted safely. Keep exact visible text in its requested script. An aspect ratio, resolution, mask, seed, or negative prompt belongs to workflow settings, not a magic phrase inside the image prompt. Return the prompt in the format the caller requests; otherwise give one copy-ready prompt and, only when useful, a separate short aspect-ratio suggestion. Do not silently invoke ComfyUI or claim a render was tested when only a prompt was written.

## This user's local workflows

- `Qwen-Image-2.1-文生图.json` is a separate ComfyUI text-to-image workflow with a connected aspect/megapixel selector, CFG 1, and 25 sampling steps. Prompt wording does not itself change the selector. Its saved example is anime, but that example is not a required character, costume, background, or style for future requests.
- `Qwen-Image-2.1-图像编辑-双图参考.json` uses the first image as the canvas and later images as references. Its saved default follows the first image's aspect ratio, caps each reference at about 1 MiP, and uses CFG 1 / 25 steps. The AstrBot `/qwen` editor currently accepts up to three images, uses `<image1>` as target and `<image2>`/`<image3>` as references, and has a configurable 1 MiP cap. Verify a different workflow's slot order instead of assuming these roles globally.
- The current AstrBot plugin implements image editing only. This skill can prepare a text-to-image prompt for the local ComfyUI workflow or a future agent, but it does **not** enable `/qwen` text-to-image by itself.

## Evidence and limits

[Qwen's official 2.1 repository](https://github.com/QwenLM/Qwen-Image-2.1) distinguishes T2I and editing prompt enhancers and documents multi-reference editing. Its [T2I rewrite prompt](https://github.com/QwenLM/Qwen-Image-2.1/blob/main/prompt_rewrite/prompts/system_prompt_t2i.txt) describes the finished frame and returns a separate ratio; its [edit rewrite prompt](https://github.com/QwenLM/Qwen-Image-2.1/blob/main/prompt_rewrite/prompts/system_prompt_edit.txt) grounds changes in the input images. The [official ComfyUI T2I](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_qwen_image_2_1_t2i.json) and [edit](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_qwen_image_2_1_image_edit.json) templates show that sizes and image slots are workflow properties. Community [T2I enhancer tests](https://www.reddit.com/r/StableDiffusion/comments/1wlvhya/qwen_image_21_pe_t2i_testing_diff_steps_mp/) and [edit examples](https://www.reddit.com/r/StableDiffusion/comments/1wlt9t9/qwen_image_21_edit_tips/) support trying fuller scene descriptions and explicit source/target roles, but are anecdotal rather than guarantees. Do not impose the official enhancer's fixed long-output format or word count on every user request.
