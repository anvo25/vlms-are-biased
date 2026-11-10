"""Step 1 of the v2 build: find the Pyllusion parameters of every published illusion image.

Usage: python match_illusions.py <v1.0 dir> <out dir> [Illusion ...]

Covers main (withtitle uses the same pictures under a title bar), original, and the
identification images of Ebbinghaus (the only illusion whose identification images are not
main's: they were drawn on a square canvas). remove_background images and Vertical-Horizontal
are recolored directly and need no parameters.
Results go to <out dir>/<Illusion>.json, keyed by "<split>/<v1 image_path>"; illusions whose
file already exists are skipped, so an interrupted run can be restarted.
"""
import glob
import io
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import pyarrow.parquet as pq
from PIL import Image

from illusions import MIN_IOU, REDRAWN, match_params


def published_images(v1_dir, split, illusion):
    out = {}
    for f in sorted(glob.glob(os.path.join(v1_dir, "data", f"{split}-*.parquet"))):
        for r in pq.read_table(f).to_pylist():
            name = r["image_path"].split("/")[-1]
            if name.startswith(illusion + "_"):
                out.setdefault(f"{split}/{r['image_path']}", r["image"]["bytes"])
    return out


def match_one(args):
    v1_dir, out_dir, illusion = args
    out_path = os.path.join(out_dir, f"{illusion}.json")
    if illusion == "Ebbinghaus" and os.path.exists(out_path) and \
            not any(k.startswith("identification/") for k in json.load(open(out_path))):
        os.remove(out_path)  # made before identification was covered
    if os.path.exists(out_path):
        return json.load(open(out_path))
    res = {}
    for split in ("main", "original") + (("identification",) if illusion == "Ebbinghaus" else ()):
        for key, data in published_images(v1_dir, split, illusion).items():
            params, score = match_params(illusion, Image.open(io.BytesIO(data)), key.split("/")[-1])
            res[key] = {"illusion": illusion, "params": params, "iou": round(float(score), 5)}
            print(f"{key}: {params} IoU={score:.4f}", flush=True)
    json.dump(res, open(out_path + ".tmp", "w"), indent=1, sort_keys=True)
    os.replace(out_path + ".tmp", out_path)
    return res


if __name__ == "__main__":
    v1_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    todo = sys.argv[3:] or REDRAWN
    cache = {}
    with ProcessPoolExecutor(len(todo)) as ex:
        for res in ex.map(match_one, [(v1_dir, out_dir, i) for i in todo]):
            cache.update(res)
    bad = {k: v for k, v in cache.items() if v["iou"] < MIN_IOU}
    print(f"\n{len(cache)} images matched; {len(bad)} below IoU {MIN_IOU}")
    for k, v in sorted(bad.items()):
        print("  ", k, v)
