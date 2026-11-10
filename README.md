# ComfyUI-OmniSVGGlue

Glue nodes that run OmniSVG 1.1 (text/image to SVG) inside a Graydient (ephemeral ComfyUI) job.

Library code (SketchDecoder, SVG tokenizer, inference helpers) comes from
[smthemex/ComfyUI_OmniSVG](https://github.com/smthemex/ComfyUI_OmniSVG) (Apache-2.0), which must sit next to this
repo in `custom_nodes/`.

- `OmniSVGLoadFast`: builds the Qwen2.5-VL architecture straight on the GPU from the bundled config (no 16 GB base
  download) and loads the OmniSVG `.bin` into it. Weights path: `models/omnisvg/<name>`.
- `OmniSVGGenerate`: text prompt, or an image URL, to SVG. Outputs a preview render (IMAGE), the SVG text (STRING)
  and the SVG packed as a pixel-data image (UTF-8 bytes in RGB, 4-byte length header).
- Self-installs missing pure-python deps and shims `cairosvg` with `resvg-py` when `libcairo` is absent.
