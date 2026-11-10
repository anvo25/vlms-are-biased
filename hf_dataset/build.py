"""Build dataset v2 from the published v1.0 parquet files.

Usage: python build.py <v1.0 dir> <v2 dir>
  (run match_illusions.py <v1.0 dir> <v2 dir>/illusion_params first)

Changes from v1.0 (rows, their order, labels and all non-illusion images stay the same):
  * optical illusions: target features drawn in red (see illusions.py) and the Q1/Q2
    questions name them ("Are the two red lines equal in length?"); remove_background
    Poggendorff keeps its v1 red and blue segments ("Are the red and blue line segments aligned?")
  * ID: new meaningful IDs (see ids.py); the v1 ID is kept in `legacy_ID`
  * topic / sub_topic: the same names in every split (main's names)
  * metadata: only the settings of the row's own task, the same key names in every split,
    plus `source_image` (the v1 file name); remove_background rows of animals, logos and
    flags get the metadata of the same sample in main
  * with_title: False instead of null (remove_background animals, logos, flags)
  * a truncated PNG in v1 identification (shoe_005_notitle_Q3_px768) is replaced by the full
    file, which main has under the same image_path (the v1 bytes are its first 327,769)
  * image_path: dropped (it was a path on the authors' machine; see metadata.source_image)
"""
import glob
import hashlib
import io
import json
import os
import re
import sys

import numpy as np
import pyarrow.parquet as pq
from datasets import Dataset, Features, Image as HFImage, Value
from PIL import Image

import ids
import illusions as il

SPLITS = ["main", "identification", "withtitle", "original", "remove_background_q1q2", "remove_background_q3"]

FEATURES = Features({
    "image": HFImage(), "ID": Value("string"), "legacy_ID": Value("string"),
    "topic": Value("string"), "sub_topic": Value("string"), "prompt": Value("string"),
    "ground_truth": Value("string"), "expected_bias": Value("string"),
    "with_title": Value("bool"), "type_of_question": Value("string"), "pixel": Value("int64"),
    "metadata": Value("string"),
})

ANSWER_YN = " Answer in curly brackets, e.g., {Yes} or {No}."
RED_QUESTIONS = {  # same wording as generators/optical_illusion_generator.py
    "ebbinghaus": ("Are the two red circles equal in size?", "Do the two red circles have the same size?"),
    "mullerlyer": ("Are the two red lines equal in length?", "Do the two red lines have the same length?"),
    "ponzo": ("Are the two red lines equal in length?", "Do the two red lines have the same length?"),
    "verticalhorizontal": ("Are the two red lines equal in length?", "Do the two red lines have the same length?"),
    "zollner": ("Are the two red lines parallel?", "Do the two red lines run parallel?"),
    "poggendorff": ("Are the two red line segments aligned?", "Do the two red line segments form a straight line?"),
}
# remove_background Poggendorff keeps its v1 red and blue segments: without the occluding bar
# the segments touch, and in one color an aligned pair would look like a single line.
RED_BLUE_QUESTIONS = ("Are the red and blue line segments aligned?",
                      "Do the red and blue line segments form a straight line?")
ILLUSION_CLASS = {v: k for k, v in ids.ILLUSION_SLUG.items()}

TOPIC = {"Optical Illusions": "Optical Illusion"}
SUB_TOPIC = {
    "Chess Board": "Chess Grid", "Go Board": "Go Grid", "Sudoku Board": "Sudoku Grid",
    "Xiangqi Board": "Xiangqi Grid",
    "Ebbinghaus Illusion": "Ebbinghaus illusion", "Müller-Lyer Illusion": "Müller-Lyer illusion",
    "Ponzo Illusion": "Ponzo illusion", "Vertical-Horizontal Illusion": "Vertical-Horizontal illusion",
    "Zöllner Illusion": "Zöllner illusion", "Poggendorff Illusion": "Poggendorff illusion",
    "animal": "Animals", "cars": "Car Logos", "shoes": "Shoe Logos",
}

# metadata: keys kept per task (main's names) and remove_background names mapped to them
KEEP = {
    "animals": ["generator"],
    "logos": ["bias_type", "element", "color", "body_type", "generator"],
    "flags": ["bias_type"],
    "pieces": ["action", "modified_square", "piece_removed", "piece_added", "piece_info", "group_id",
               "circles_count", "triangles_count"],
    "boards": ["action", "dimension_modified", "original_dimensions", "new_dimensions", "position",
               "remove_index", "insert_index", "visual_cue"],
    "grid": ["action", "modified_cell", "grid_type", "grid_size", "standard_count", "modified_count",
             "pattern_type", "target_shape_index", "group_id", "type", "color"],
    "illusion": ["illusion_type", "strength", "size_min", "diff", "is_diff_zero"],
}
RENAME = {
    "action_type": "action", "modified_square_algebraic": "modified_square",
    "anomalous_cell_algebraic": "modified_cell", "anomaly_target_shape_index": "target_shape_index",
    "anomaly_replaced_with_shape": "type", "modified_count_in_cell": "modified_count",
    "standard_count_in_cell": "standard_count", "pattern_id_used": "pattern_type",
    "illusion_type_key": "illusion_type", "difference_value": "diff",
    "is_difference_zero": "is_diff_zero",
}


def load_meta(s):
    return ids.metadata({"metadata": s})


def clean_metadata(row, md):
    task = row["ID"].split("-")[0]
    md = {RENAME.get(k, k): v for k, v in md.items()}
    out = {}
    for k in KEEP[task]:
        v = md.get(k)
        if v in (None, "", [], {}, "N/A"):
            continue
        action = str(md.get("action", ""))
        if task == "boards" and ((k == "remove_index" and "remove" not in action) or
                                 (k == "insert_index" and "add" not in action)):
            continue  # only the index the action uses
        if task == "grid" and k in ("type", "color") and "replace" not in str(md.get("action", "")):
            continue  # replacement shape, only for "replace"
        if task == "illusion" and k in ("strength", "size_min") and \
                (k == "strength") == (md.get("illusion_type") == "VerticalHorizontal"):
            continue  # Vertical-Horizontal uses size_min, the others strength
        if task == "pieces" and isinstance(v, (int, float)) and v == 0 and k != "group_id":
            continue
        if k == "body_type":
            v = str(v).removesuffix(".png")
        if k == "grid_type":
            v = str(v).removesuffix("_pattern")
        out[k] = v
    out["source_image"] = row["image_path"].split("/")[-1]
    return out


def read_split(v1_dir, split, columns=None):
    files = sorted(glob.glob(os.path.join(v1_dir, "data", f"{split}-*.parquet")))
    return [r for f in files for r in pq.read_table(f, columns=columns).to_pylist()]


def decodes(data):
    try:
        Image.open(io.BytesIO(data)).convert("RGB")
        return True
    except OSError:
        return False


def img_hash(data):
    """Hash of the decoded pixels, so the same picture matches however it was encoded."""
    im = Image.open(io.BytesIO(data)).convert("RGB")
    return hashlib.md5(np.asarray(im).tobytes() + str(im.size).encode()).hexdigest()


class IllusionImages:
    """New (red) image bytes for illusion rows, cached by the old image's content."""

    def __init__(self, params_dir):
        self.params = {}
        for f in glob.glob(os.path.join(params_dir, "*.json")):
            self.params.update(json.load(open(f)))
        self.by_hash = {}       # old image md5 -> params entry (filled from main/original)
        self.cache = {}         # old image md5 -> new png bytes
        self.log = []           # (ID, method, iou)

    def register(self, split, row):
        p = self.params.get(f"{split}/{row['image_path']}")
        if p:
            self.by_hash[img_hash(row["image"]["bytes"])] = p

    def red(self, row, method_hint):
        data = row["image"]["bytes"]
        h = img_hash(data)
        if h in self.cache:
            return self.cache[h]
        old = Image.open(io.BytesIO(data)).convert("RGB")
        ill = ILLUSION_CLASS[row["ID"].split("-")[1]]
        if method_hint == "recolor" or ill not in il.REDRAWN:
            new, method, score = il.recolor_red(old), "recolor", 1.0
        else:
            p = self.by_hash.get(h)
            if p is None:
                raise RuntimeError(f"no Pyllusion parameters for {row['legacy_ID']}")
            new = il.render(ill, p["params"], old.size)
            score = il.iou(il.ink(old), il.ink(new))
            if score < il.MIN_IOU:
                raise RuntimeError(f"redraw of {row['legacy_ID']} differs from v1 (IoU {score:.4f})")
            method = "redraw"
        self.log.append((row["ID"], method, round(float(score), 5)))
        self.cache[h] = il.png_bytes(new)
        return self.cache[h]

    def params_for(self, row):
        p = self.by_hash.get(img_hash(row["image"]["bytes"]))
        return p["params"] if p else None


def titled(row, notitle_old_bytes, notitle_new_bytes):
    """Replace the picture under a withtitle image's title bar with the new picture."""
    old = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
    body_old = Image.open(io.BytesIO(notitle_old_bytes)).convert("RGB")
    body_new = Image.open(io.BytesIO(notitle_new_bytes)).convert("RGB")
    bar = old.height - body_old.height
    if old.width != body_old.width or bar <= 0:
        raise RuntimeError(f"{row['legacy_ID']}: title layout does not match its notitle image")
    score = il.iou(il.ink(old.crop((0, bar, old.width, old.height))), il.ink(body_old))
    if score < il.MIN_IOU:
        raise RuntimeError(f"{row['legacy_ID']}: picture under the title differs from main (IoU {score:.4f})")
    out = old.copy()
    out.paste(body_new, (0, bar))
    return il.png_bytes(out)


def main(v1_dir, v2_dir):
    os.makedirs(os.path.join(v2_dir, "data"), exist_ok=True)
    cols = ["ID", "image_path", "topic", "sub_topic", "prompt", "type_of_question", "pixel", "metadata"]
    table = []
    for split in SPLITS:
        for i, r in enumerate(read_split(v1_dir, split, cols)):
            r.update(split=split, row=i)
            table.append(r)
    ids.assign_ids(table)
    new_id = {(r["split"], r["row"]): (r["ID"], r["legacy_ID"]) for r in table}

    # main metadata of animals/logos/flags, to fill the same samples in remove_background
    main_meta = {}
    for r in table:
        if r["split"] == "main" and r["ID"].split("-")[0] in ("animals", "logos", "flags"):
            main_meta["-".join(r["ID"].split("-")[:4])] = load_meta(r["metadata"])

    images = IllusionImages(os.path.join(v2_dir, "illusion_params"))
    main_bytes = {}  # v1 image_path -> image bytes in main, to repair truncated copies elsewhere
    notitle_images = {}  # new notitle ID (minus condition) -> (old bytes, new bytes), for withtitle
    changes = []
    for split in ["main", "original", "identification", "withtitle", "remove_background_q1q2", "remove_background_q3"]:
        rows = read_split(v1_dir, split)
        if split in ("main", "original", "identification"):
            for r in rows:
                images.register(split, r)
        if split == "main":
            main_bytes = {r["image_path"]: r["image"]["bytes"] for r in rows}
        out = []
        for i, r in enumerate(rows):
            r["ID"], r["legacy_ID"] = new_id[(split, i)]
            task, sub = r["ID"].split("-")[:2]
            before = {k: r[k] for k in ("prompt", "topic", "sub_topic")}
            md = load_meta(r["metadata"])
            if split.startswith("remove_background") and not md:
                md = main_meta.get("-".join(r["ID"].split("-")[:4]), {})
            r["metadata"] = clean_metadata(r, md)
            if r["with_title"] is None:  # remove_background animals/logos/flags: no title
                r["with_title"] = False
            r["topic"] = TOPIC.get(r["topic"], r["topic"])
            r["sub_topic"] = SUB_TOPIC.get(r["sub_topic"], r["sub_topic"])
            if split == "original" and r["sub_topic"] == "flags":
                r["sub_topic"] = "Flags 2D Stars" if sub == "stars" else "Flags 2D Stripes"
            image_changed = False
            red_blue = task == "illusion" and sub == "poggendorff" and split.startswith("remove_background")
            if red_blue:
                q = r["type_of_question"]
                if q in ("Q1", "Q2"):
                    r["prompt"] = RED_BLUE_QUESTIONS[int(q[1]) - 1] + ANSWER_YN
                r["metadata"]["targets"] = "red and blue"
                r["image"] = {"bytes": r["image"]["bytes"], "path": None}
            elif task == "illusion":
                q = r["type_of_question"]
                if q in ("Q1", "Q2"):
                    r["prompt"] = RED_QUESTIONS[sub][int(q[1]) - 1] + ANSWER_YN
                r["prompt"] = r["prompt"].replace("Vertical–Horizontal", "Vertical-Horizontal")
                key = "-".join(p for j, p in enumerate(r["ID"].split("-")) if j not in (4, 5))  # no condition, question
                if split == "withtitle":
                    old_b, new_b = notitle_images[key]
                    new_bytes = titled(r, old_b, new_b)
                else:
                    new_bytes = images.red(r, "recolor" if split.startswith("remove_background") else "auto")
                    if split == "main":
                        notitle_images[key] = (r["image"]["bytes"], new_bytes)
                    params = images.params_for(r) if split in ("main", "original", "identification") else None
                    if params and sub != "verticalhorizontal":
                        r["metadata"]["pyllusion_params"] = params
                r["metadata"]["targets"] = "red"
                r["image"] = {"bytes": new_bytes, "path": None}
                image_changed = True
            else:
                data = r["image"]["bytes"]
                if not decodes(data):
                    full = main_bytes.get(r["image_path"], b"")
                    if not (full.startswith(data) and decodes(full)):
                        raise RuntimeError(f"{r['legacy_ID']}: image does not decode and cannot be repaired")
                    data, image_changed = full, True
                    print(f"  repaired truncated image of {r['legacy_ID']} from main", flush=True)
                r["image"] = {"bytes": data, "path": None}
            changed = [k for k in before if before[k] != r[k]]
            changes.append({"split": split, "row": i, "ID": r["ID"], "legacy_ID": r["legacy_ID"],
                            "image_changed": image_changed, "changed": ",".join(changed)})
            r["metadata"] = json.dumps(r["metadata"], ensure_ascii=False, sort_keys=True)
            r.pop("image_path")
            out.append({k: r[k] for k in FEATURES})
        ds = Dataset.from_list(out, features=FEATURES)
        # Like v1: dictionary encoding in small row groups, so rows sharing a picture (Q1/Q2)
        # store it once.
        pq.write_table(ds.data.table, os.path.join(v2_dir, "data", f"{split}.parquet"),
                       row_group_size=100, use_dictionary=True, compression="snappy",
                       dictionary_pagesize_limit=256 * 1024 * 1024)
        print(f"{split}: {len(ds)} rows, {sum(c['image_changed'] for c in changes if c['split'] == split)} images changed", flush=True)
        del rows, out, ds
    json.dump(images.log, open(os.path.join(v2_dir, "illusion_images.json"), "w"))
    with open(os.path.join(v2_dir, "changes.jsonl"), "w") as f:
        for c in changes:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
