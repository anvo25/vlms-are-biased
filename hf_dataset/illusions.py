"""Red-target versions of the published optical-illusion images (dataset v2).

The v1.0 illusion images were drawn with Pyllusion with every feature in black (Ebbinghaus
already had red target circles). v2 shows the features the questions ask about in red,
without changing anything else in the picture:

* main / withtitle / original: each published image is redrawn with Pyllusion 1.4 at its
  published size with Pyllusion's default colors, which draw the target features red. The
  parameters are found by search, because the file names do not always give the ones that
  were used (e.g., most images were drawn with the opposite strength sign). A redraw is
  accepted only if its shapes match the published image pixel for pixel (`MIN_IOU`).
* remove_background, and Vertical-Horizontal in every split: these images contain only the
  target features, so every dark pixel is recolored to the same red, keeping the
  anti-aliasing. (The published Vertical-Horizontal inverted T was not drawn by Pyllusion,
  whose version is two separate horizontal lines, so it cannot be redrawn.)

withtitle images keep their published title bar; only the picture under it is replaced.
"""
import io
import itertools
import re

import numpy as np
import pyllusion
from PIL import Image

ILLUSIONS = ["Ebbinghaus", "MullerLyer", "Ponzo", "VerticalHorizontal", "Zollner", "Poggendorff"]
REDRAWN = ["Ebbinghaus", "MullerLyer", "Ponzo", "Zollner", "Poggendorff"]  # the others are recolored
RED = np.array([255, 0, 0])
MIN_IOU = 0.999  # shapes of the redraw vs the published image (dark pixels)
INK = 160        # a pixel is "dark" if any channel is below this


def ink(img):
    return (np.asarray(img.convert("RGB")).astype(int) < INK).any(-1)


def iou(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


def file_params(name):
    """Parameters encoded in a v1 file name, e.g. Ponzo_013_str15_diff0p15_... -> (15, 0.15)."""
    num = lambda m: (-1 if m.group(1) else 1) * float(m.group(2).replace("p", "."))
    d = num(re.search(r"_diff(neg)?([0-9p]+)", name))
    m = re.search(r"_min([0-9p]+)", name)
    if m:
        return {"size_min": float(m.group(1).replace("p", ".")), "difference": d}
    return {"illusion_strength": num(re.search(r"_str(neg)?([0-9p]+)", name)), "difference": d}


def render(illusion, params, size):
    """Pyllusion 1.4 with its default (red-target) colors, at the given (width, height)."""
    return getattr(pyllusion, illusion)(**params).to_image(width=size[0], height=size[1]).convert("RGB")


def _candidates(illusion, p):
    """File parameters with sign flips first, then a wider grid."""
    if "size_min" in p:
        key, val = "size_min", p["size_min"]
        grid = [round(x * 0.1, 1) for x in range(1, 16)]
    else:
        key, val = "illusion_strength", p["illusion_strength"]
        grid = [s * g for s in range(1, 91) for g in (1, -1)]
    diffs = [p["difference"], -p["difference"]] + [
        round(d * g, 2) for d in (0, .05, .1, .15, .2, .25, .3, .35, .4, .5, .6, .7, .8, 1, 1.5, 2, 2.5, 3, 3.5) for g in (1, -1)]
    first = [val, -val] if key != "size_min" else [val]
    seen = set()
    for v, d in itertools.chain(itertools.product(first, diffs[:2]), itertools.product(first + grid, diffs)):
        if (v, d) not in seen:
            seen.add((v, d))
            yield {key: v, "difference": d}


def match_params(illusion, old_img, name):
    """Find the Pyllusion parameters that redraw `old_img` exactly. Returns (params, iou)."""
    a = ink(old_img)
    best = (None, -1.0)
    for params in _candidates(illusion, file_params(name)):
        try:
            score = iou(a, ink(render(illusion, params, old_img.size)))
        except Exception:
            continue
        if score > best[1]:
            best = (params, score)
        if score >= MIN_IOU:
            break
    return best


def recolor_red(img):
    """Recolor every dark pixel red, keeping anti-aliasing (for images with only target features)."""
    a = np.asarray(img.convert("RGB")).astype(float)
    darkness = 1 - a.min(-1, keepdims=True) / 255.0          # 0 = white, 1 = full ink
    out = 255 * (1 - darkness) + RED * darkness
    return Image.fromarray(out.round().clip(0, 255).astype(np.uint8))


def png_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
