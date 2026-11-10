"""ComfyUI-OmniSVGGlue: fast loader + text/image-to-SVG runner for OmniSVG 1.1 on top of smthemex/ComfyUI_OmniSVG."""
import os
import subprocess

# Graydient keeps the FIRST clone of a requirements.github repo per project dir; pull so pushes reach it.
try:
    subprocess.run(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "pull", "--ff-only", "-q"],
                   timeout=20, capture_output=True)
except Exception:
    pass

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: E402

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
