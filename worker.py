"""Subprocess entry: runs OmniSVG load+generate under whatever transformers is first on PYTHONPATH."""
import importlib.util
import json
import os
import sys

job = json.load(open(sys.argv[1], encoding="utf-8"))
spec = importlib.util.spec_from_file_location("omnisvg_glue_nodes", job["nodes_path"],
                                              submodule_search_locations=None)
m = importlib.util.module_from_spec(spec)
sys.modules["omnisvg_glue_nodes"] = m
spec.loader.exec_module(m)
import transformers
print(f"[worker] transformers {transformers.__version__}", flush=True)
(model,) = m.OmniSVGLoadFast().load(**job["load"])
prev, svg, _data, sheet = m.OmniSVGGenerate().run(model, **job["gen"])
from PIL import Image
import numpy as np
def save(t, name):
    a = (t[0].cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
    Image.fromarray(a).save(os.path.join(job["out"], name))
save(prev, "preview.png")
save(sheet, "sheet.png")
open(os.path.join(job["out"], "svg.svg"), "w", encoding="utf-8").write(svg)
print("[worker] done", flush=True)
