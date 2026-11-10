"""Check dataset v2 against v1.0 before upload.

Usage: python check.py <v1.0 dir> <v2 dir>

Fails (exit 1) unless, for every split:
  * the rows are the same rows in the same order (legacy_ID == v1 ID), with the same
    ground_truth, expected_bias, with_title, type_of_question and pixel
  * IDs are unique across the whole dataset and have 7 "-"-separated fields
  * images of non-illusion rows are pixel-identical to v1 (except a v1 image that is a
    truncated PNG, whose v2 image must be the full file that starts with the same bytes)
  * illusion images have the same size and the same shapes as v1 (dark pixels, IoU >= 0.999),
    show red, and withtitle images keep their v1 title bar exactly
  * prompts changed only for illusion rows
  * main's topic values (used by lmms-eval for per-topic accuracy) are unchanged
"""
import glob
import io
import json
import os
import sys
from collections import Counter

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

import illusions as il

SPLITS = ["main", "identification", "withtitle", "original", "remove_background_q1q2", "remove_background_q3"]
SAME = ["ground_truth", "expected_bias", "with_title", "type_of_question", "pixel"]
COLUMNS = ["image", "ID", "legacy_ID", "topic", "sub_topic", "prompt", "ground_truth", "expected_bias",
           "with_title", "type_of_question", "pixel", "metadata"]


def decodes(data):
    try:
        Image.open(io.BytesIO(data)).convert("RGB")
        return True
    except OSError:
        return False


def rgb(data):
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


def main(v1_dir, v2_dir):
    errors, notes, all_ids = [], Counter(), Counter()
    for split in SPLITS:
        old = [r for f in sorted(glob.glob(os.path.join(v1_dir, "data", f"{split}-*.parquet")))
               for r in pq.read_table(f).to_pylist()]
        new_t = pq.read_table(os.path.join(v2_dir, "data", f"{split}.parquet"))
        if new_t.column_names != COLUMNS:
            errors.append(f"{split}: columns {new_t.column_names}")
        new = new_t.to_pylist()
        if len(old) != len(new):
            errors.append(f"{split}: {len(old)} rows in v1, {len(new)} in v2")
            continue
        for o, n in zip(old, new):
            tag = f"{split}/{n['ID']}"
            all_ids[n["ID"]] += 1
            if n["legacy_ID"] != o["ID"]:
                errors.append(f"{tag}: legacy_ID {n['legacy_ID']} != v1 ID {o['ID']}")
            for k in SAME:
                if o[k] != n[k] and not (k == "with_title" and o[k] is None and n[k] is False):
                    errors.append(f"{tag}: {k} {o[k]!r} -> {n[k]!r}")
            if len(n["ID"].split("-")) != 7:
                errors.append(f"{tag}: ID does not have 7 fields")
            json.loads(n["metadata"])
            is_ill = n["ID"].startswith("illusion-")
            if o["prompt"] != n["prompt"]:
                notes[f"{split}: prompt changed"] += 1
                if not is_ill:
                    errors.append(f"{tag}: prompt changed for a non-illusion row")
            if split == "main" and o["topic"] != n["topic"]:
                errors.append(f"{tag}: main topic {o['topic']!r} -> {n['topic']!r}")
            if not decodes(n["image"]["bytes"]):
                errors.append(f"{tag}: v2 image does not decode")
                continue
            if not decodes(o["image"]["bytes"]):
                if is_ill or not n["image"]["bytes"].startswith(o["image"]["bytes"]):
                    errors.append(f"{tag}: v1 image does not decode and v2 is not its full version")
                notes[f"{split}: truncated v1 image repaired"] += 1
                continue
            a, b = rgb(o["image"]["bytes"]), rgb(n["image"]["bytes"])
            if not is_ill:
                if a.shape != b.shape or not np.array_equal(a, b):
                    errors.append(f"{tag}: image changed for a non-illusion row")
                continue
            if split.startswith("remove_background") and n["ID"].startswith("illusion-poggendorff-"):
                b_ = rgb(n["image"]["bytes"])
                blue = ((b_[..., 2] > 200) & (b_[..., 0] < 60) & (b_[..., 1] < 60)).sum()
                red_ = ((b_[..., 0] > 200) & (b_[..., 1] < 60) & (b_[..., 2] < 60)).sum()
                if o["image"]["bytes"] != n["image"]["bytes"] or not (blue and red_):
                    errors.append(f"{tag}: remove_background Poggendorff must keep its v1 red/blue image")
                notes[f"{split}: Poggendorff red/blue image kept"] += 1
                continue
            notes[f"{split}: illusion image redrawn/recolored"] += 1
            if a.shape != b.shape:
                errors.append(f"{tag}: image size {a.shape} -> {b.shape}")
                continue
            score = il.iou((a < il.INK).any(-1), (b < il.INK).any(-1))
            if score < il.MIN_IOU:
                errors.append(f"{tag}: shapes differ from v1 (IoU {score:.4f})")
            red = (b[..., 0] > 200) & (b[..., 1] < 60) & (b[..., 2] < 60)
            if red.sum() == 0:
                errors.append(f"{tag}: no red pixels")
            if split == "withtitle":
                bar = a.shape[0] - int(n["pixel"])
                if bar > 0 and not np.array_equal(a[:bar], b[:bar]):
                    errors.append(f"{tag}: title bar changed")
        notes[f"{split}: rows"] = len(new)
    dup = [i for i, c in all_ids.items() if c > 1]
    if dup:
        errors.append(f"{len(dup)} duplicate IDs, e.g. {dup[:3]}")
    for k, v in sorted(notes.items()):
        print(f"  {k}: {v}")
    print(f"{sum(all_ids.values())} rows, {len(all_ids)} unique IDs")
    if errors:
        print(f"\nFAILED: {len(errors)} problems")
        for e in errors[:50]:
            print("  ", e)
        sys.exit(1)
    print("\nOK: all checks passed")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
