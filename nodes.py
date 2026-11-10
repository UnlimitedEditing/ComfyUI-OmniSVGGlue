import contextlib
import ctypes.util
import gc
import importlib
import importlib.util
import io
import math
import os
import random
import struct
import subprocess
import sys
import types
import urllib.request

import numpy as np
import torch
from PIL import Image

import folder_paths

TAG = "[OmniSVGGlue]"
_CACHE = {}


# ---------------------------------------------------------------------------------------------------------------
# dependency handling (requirements.pip is not reliably executed on Graydient, so self-install + shim)
# ---------------------------------------------------------------------------------------------------------------
def _pip(*specs):
    cmd = [sys.executable, "-m", "pip", "install", "--no-input", "--disable-pip-version-check", "-q", *specs]
    print(f"{TAG} pip install {' '.join(specs)}", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"{TAG} pip failed ({r.returncode}): {r.stderr[-400:]}", flush=True)
    return r.returncode == 0


def _have(mod):
    try:
        return importlib.util.find_spec(mod) is not None
    except Exception:
        return False


def _stub_module(name, attrs=None):
    m = types.ModuleType(name)
    m.__dict__["__stub__"] = True
    for k, v in (attrs or {}).items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


def _install_cairosvg_shim():
    """Replace cairosvg with a resvg-py backed shim exposing svg2png(bytestring=..., output_width=, output_height=)."""
    if not _have("resvg_py"):
        _pip("resvg-py")
    import resvg_py

    def svg2png(bytestring=None, url=None, output_width=None, output_height=None, **kw):
        if bytestring is None:
            raise ValueError("shim svg2png needs bytestring")
        svg = bytestring.decode("utf-8") if isinstance(bytestring, (bytes, bytearray)) else str(bytestring)
        args = {"svg_string": svg}
        if output_width:
            args["width"] = int(output_width)
        if output_height:
            args["height"] = int(output_height)
        out = resvg_py.svg_to_bytes(**args)
        return bytes(out)

    _stub_module("cairosvg", {"svg2png": svg2png})
    print(f"{TAG} cairosvg unavailable: using resvg-py shim", flush=True)


def _ensure_deps():
    if _CACHE.get("deps_ok"):
        return
    # cairosvg: real one needs the libcairo system library
    try:
        import cairosvg  # noqa: F401
        cairosvg.svg2png(bytestring=b'<svg xmlns="http://www.w3.org/2000/svg" width="8" height="8"/>')
    except Exception:
        sys.modules.pop("cairosvg", None)
        ok = False
        if ctypes.util.find_library("cairo"):
            _pip("cairosvg", "cairocffi")
            try:
                import cairosvg  # noqa: F401
                cairosvg.svg2png(bytestring=b'<svg xmlns="http://www.w3.org/2000/svg" width="8" height="8"/>')
                ok = True
            except Exception:
                sys.modules.pop("cairosvg", None)
        if not ok:
            _install_cairosvg_shim()

    missing = [pip for mod, pip in (("accelerate", "accelerate"), ("networkx", "networkx"), ("cv2", None),
                                    ("yaml", "pyyaml"), ("einops", "einops"), ("shapely", "shapely")) if pip and not _have(mod)]
    if missing:
        _pip(*missing)
    if not _have("cv2"):
        _pip("--no-deps", "opencv-python-headless")

    if not _have("moviepy"):  # deepsvg imports it unconditionally but generation never uses it
        _stub_module("moviepy", {"ImageClip": None, "concatenate_videoclips": None})

    if not _have("IPython"):
        if not _pip("ipython"):
            _stub_module("IPython")
            _stub_module("IPython.display")
            sys.modules["IPython"].display = sys.modules["IPython.display"]

    try:
        import qwen_vl_utils  # noqa: F401
    except Exception:
        if not _pip("qwen-vl-utils"):
            def process_vision_info(messages, *a, **k):
                imgs = []
                for m in messages:
                    c = m.get("content")
                    if isinstance(c, list):
                        imgs += [x["image"] for x in c if isinstance(x, dict) and x.get("type") == "image"]
                return (imgs or None), None
            _stub_module("qwen_vl_utils", {"process_vision_info": process_vision_info})
            print(f"{TAG} qwen_vl_utils unavailable: using minimal shim", flush=True)
    _CACHE["deps_ok"] = True


def _find_repo():
    for base in folder_paths.get_folder_paths("custom_nodes"):
        for name in ("ComfyUI_OmniSVG", "ComfyUI-OmniSVG"):
            p = os.path.join(base, name)
            if os.path.isfile(os.path.join(p, "OmniSVG", "inference.py")):
                return p
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = os.path.join(here, "ComfyUI_OmniSVG")
    if os.path.isfile(os.path.join(p, "OmniSVG", "inference.py")):
        return p
    raise RuntimeError("smthemex/ComfyUI_OmniSVG repo not found in custom_nodes (add it to requirements.github)")


def _inference():
    if "inf" in _CACHE:
        return _CACHE["inf"], _CACHE["pkg"]
    _ensure_deps()
    root = _find_repo()
    if root not in sys.path:
        sys.path.insert(0, root)
    inf = importlib.import_module("OmniSVG.inference")
    _CACHE["inf"], _CACHE["pkg"] = inf, os.path.join(root, "OmniSVG")
    return inf, _CACHE["pkg"]


def _render(svg_str, w, h):
    import cairosvg
    png = cairosvg.svg2png(bytestring=svg_str.encode("utf-8"), output_width=w, output_height=h)
    rgba = Image.open(io.BytesIO(png)).convert("RGBA")
    bg = Image.new("RGB", rgba.size, (255, 255, 255))
    bg.paste(rgba, mask=rgba.split()[3])
    return bg


def _pil2tensor(img):
    return torch.from_numpy(np.array(img.convert("RGB")).astype(np.float32) / 255.0).unsqueeze(0)


def _encode_string_as_image(text):
    """Same convention as the ForgeExpress data image: 4-byte big-endian length + UTF-8, 3 bytes per pixel."""
    raw = text.encode("utf-8")
    payload = struct.pack(">I", len(raw)) + raw
    while len(payload) % 3:
        payload += b"\x00"
    side = max(1, math.ceil(math.sqrt(len(payload) // 3)))
    padded = payload + b"\x00" * (side * side * 3 - len(payload))
    arr = np.frombuffer(padded, dtype=np.uint8).reshape(side, side, 3).astype(np.float32) / 255.0
    return torch.from_numpy(arr.copy()).unsqueeze(0)


def _download_image(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return Image.open(io.BytesIO(r.read()))


# ---------------------------------------------------------------------------------------------------------------
# nodes
# ---------------------------------------------------------------------------------------------------------------
class OmniSVGLoadFast:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "weights_name": ("STRING", {"default": "OmniSVG1.1_8B.bin"}),
            "model_size": (["8B", "4B"], {"default": "8B"}),
        }}

    RETURN_TYPES = ("OMNISVG_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "OmniSVG"

    def load(self, weights_name, model_size):
        import yaml
        from transformers import AutoConfig, AutoProcessor, AutoTokenizer
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as QwenVL
        except Exception:
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration as QwenVL

        inf, pkg = _inference()
        with open(os.path.join(pkg, "config.yaml"), "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        local_repo = os.path.join(pkg, "Qwen2.5-VL-7B-Instruct" if model_size == "8B" else "Qwen2.5-VL-3B-Instruct")
        weights = os.path.join(folder_paths.models_dir, "omnisvg", weights_name)
        if not os.path.isfile(weights):
            raise FileNotFoundError(f"OmniSVG weights not found: {weights}")

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bf16 = torch.bfloat16 if dev.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
        print(f"{TAG} loading tokenizer/processor from {local_repo}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(local_repo, padding_side="left", trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(local_repo, padding_side="left", trust_remote_code=True)
        processor.tokenizer.padding_side = "left"

        mcfg = config["models"][model_size]["model"]
        vocab = mcfg["vocab_size"]
        cfg = AutoConfig.from_pretrained(local_repo)
        cfg.vocab_size = vocab
        if getattr(cfg, "text_config", None) is not None:
            cfg.text_config.vocab_size = vocab

        print(f"{TAG} building {model_size} architecture on {dev} ({bf16}), vocab {vocab}", flush=True)
        try:
            from transformers.modeling_utils import no_init_weights
            ctx = no_init_weights()
        except Exception:
            ctx = contextlib.nullcontext()
        prev = torch.get_default_dtype()
        torch.set_default_dtype(bf16)
        try:
            with ctx, torch.device(dev):
                model = QwenVL._from_config(cfg, attn_implementation="sdpa")
        finally:
            torch.set_default_dtype(prev)
        model = model.to(dtype=bf16)

        print(f"{TAG} reading {weights}", flush=True)
        try:
            sd = torch.load(weights, map_location="cpu", weights_only=True, mmap=True)
        except Exception as e:
            print(f"{TAG} mmap load failed ({e}); plain load", flush=True)
            sd = torch.load(weights, map_location="cpu", weights_only=True)

        has_lm = any(k.startswith("model.language_model.") for k in model.state_dict().keys())
        new = {}
        for k, v in sd.items():
            if k.startswith("transformer."):
                k = k[len("transformer."):]
            if has_lm:
                if k.startswith("visual."):
                    k = "model." + k
                elif k.startswith(("model.layers.", "model.embed_tokens.", "model.norm.")):
                    k = k.replace("model.", "model.language_model.", 1)
            new[k] = v
        missing, unexpected = model.load_state_dict(new, strict=False)
        missing = [k for k in missing if "inv_freq" not in k]
        print(f"{TAG} load_state_dict: {len(new)} tensors, missing {len(missing)}, unexpected {len(unexpected)}", flush=True)
        if unexpected:
            print(f"{TAG} unexpected sample: {unexpected[:5]}", flush=True)
        if missing:
            raise RuntimeError(f"OmniSVG weights did not fill the model; missing sample: {missing[:8]}")
        del sd, new
        gc.collect()

        m = config["model"]
        model.config.bos_token_id = m["bos_token_id"]
        model.config.eos_token_id = m["eos_token_id"]
        model.config.pad_token_id = m["pad_token_id"]
        model.eval()
        decoder = types.SimpleNamespace(transformer=model)

        from OmniSVG.tokenizer import SVGTokenizer
        svg_tokenizer = SVGTokenizer(os.path.join(pkg, "config.yaml"), model_size=model_size)
        torch.cuda.empty_cache()
        return ({"tokenizer": tokenizer, "processor": processor, "sketch_decoder": decoder,
                 "svg_tokenizer": svg_tokenizer, "config": config, "torch_dtype": bf16, "dev": dev},)


class OmniSVGGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("OMNISVG_MODEL",),
            "prompt": ("STRING", {"multiline": True, "default": "A yellow t-shirt with a heart design."}),
            "init_image_url": ("STRING", {"default": ""}),
            "max_length": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
            "temperature": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.5, "step": 0.05}),
            "width": ("INT", {"default": 0, "min": 0, "max": 4096}),
            "height": ("INT", {"default": 0, "min": 0, "max": 4096}),
            "init_image_filename": ("STRING", {"default": ""}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 2 ** 31 - 1}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING", "IMAGE")
    RETURN_NAMES = ("preview", "svg_text", "svg_data")
    FUNCTION = "run"
    CATEGORY = "OmniSVG"

    def run(self, model, prompt, init_image_url, max_length, temperature, width, height, init_image_filename, seed):
        inf, _ = _inference()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        out_dir = os.path.join(folder_paths.get_output_directory(), "omnisvg")
        os.makedirs(out_dir, exist_ok=True)
        args = types.SimpleNamespace(
            verbose=True, output=out_dir, save_all_candidates=False, save_png=True,
            top_p=None, top_k=None, repetition_penalty=None,
            temperature=(temperature if temperature > 0 else None),
            max_length=max_length, replace_background=False, num_candidates=1, use_accelerate=True)
        dev, dt = model["dev"], model["torch_dtype"]
        url = (init_image_url or "").strip()
        fname = (init_image_filename or "").strip()
        local = None
        if not url and fname:
            for cand in (fname, os.path.join(folder_paths.get_input_directory(), fname)):
                if os.path.isfile(cand):
                    local = cand
                    break
            if local is None:
                print(f"{TAG} init_image_filename '{fname}' not found in input dir; falling back to text mode", flush=True)
        if url or local:
            print(f"{TAG} image-to-SVG from {url[:80] or local}", flush=True)
            img = _download_image(url) if url else Image.open(local)
            files, _pngs = inf.process_image_to_svg(model["sketch_decoder"], model["svg_tokenizer"], model["processor"],
                                                    [img], dev, dt, args, model["config"])
        else:
            print(f"{TAG} text-to-SVG: {prompt[:80]}", flush=True)
            files, _pngs = inf.process_text_to_svg(model["sketch_decoder"], model["svg_tokenizer"], model["processor"],
                                                   prompt, dev, dt, args, "auto", model["config"])
        paths = []
        for f in files:
            if isinstance(f, str):
                paths.append(f)
            elif isinstance(f, list):
                paths += [x for x in f if isinstance(x, str)]
        if not paths:
            raise RuntimeError("OmniSVG produced no valid SVG for this input. Try another seed, a simpler prompt, "
                               "or a larger max_length.")
        with open(paths[0], "r", encoding="utf-8") as fh:
            svg = fh.read()
        w = width or 1024
        h = height or 1024
        preview = _render(svg, w, h)
        gc.collect()
        torch.cuda.empty_cache()
        print(f"{TAG} svg {len(svg)} chars, saved {paths[0]}", flush=True)
        return (_pil2tensor(preview), svg, _encode_string_as_image(svg))


NODE_CLASS_MAPPINGS = {"OmniSVGLoadFast": OmniSVGLoadFast, "OmniSVGGenerate": OmniSVGGenerate}
NODE_DISPLAY_NAME_MAPPINGS = {"OmniSVGLoadFast": "OmniSVG Load (fast)", "OmniSVGGenerate": "OmniSVG Generate"}
