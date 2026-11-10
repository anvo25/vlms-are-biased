"""Make a local review page and spreadsheet comparing dataset v1.0 and v2.

Usage: python review.py <v1.0 dir> <v2 dir> <out dir>
Then:  cd <out dir> && python -m http.server 8000   and open http://localhost:8000

Writes:
  index.html   every changed image side by side (v1 | v2) with old/new ID and prompt, filterable,
               plus a few unchanged rows per task to check the new IDs and metadata
  rows.csv     every row of every split: old and new ID, prompt, metadata, labels, what changed
  img/         the images the page shows
"""
import glob
import html
import io
import json
import os
import sys

import pandas as pd
import pyarrow.parquet as pq
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True  # v1 has one truncated PNG; show what is there

SPLITS = ["main", "identification", "withtitle", "original", "remove_background_q1q2", "remove_background_q3"]


def save_thumb(data, path, height=260):
    im = Image.open(io.BytesIO(data)).convert("RGB")
    im.thumbnail((int(height * 2.2), height))
    im.save(path)


def main(v1_dir, v2_dir, out_dir):
    os.makedirs(os.path.join(out_dir, "img"), exist_ok=True)
    table, cards = [], []
    for split in SPLITS:
        old = [r for f in sorted(glob.glob(os.path.join(v1_dir, "data", f"{split}-*.parquet")))
               for r in pq.read_table(f).to_pylist()]
        new = pq.read_table(os.path.join(v2_dir, "data", f"{split}.parquet")).to_pylist()
        shown_per_task = {}
        for i, (o, n) in enumerate(zip(old, new)):
            image_changed = o["image"]["bytes"] != n["image"]["bytes"]
            changed = [k for k in ("prompt", "topic", "sub_topic") if o[k] != n[k]]
            if image_changed:
                changed.insert(0, "image")
            table.append({
                "split": split, "row": i, "ID": n["ID"], "legacy_ID": o["ID"], "changed": ",".join(changed),
                "topic": n["topic"], "sub_topic": n["sub_topic"], "type_of_question": n["type_of_question"],
                "pixel": n["pixel"], "ground_truth": n["ground_truth"], "expected_bias": n["expected_bias"],
                "prompt_v2": n["prompt"], "prompt_v1": o["prompt"] if o["prompt"] != n["prompt"] else "",
                "metadata_v2": n["metadata"], "metadata_v1": o["metadata"], "image_path_v1": o["image_path"],
            })
            task = "-".join(n["ID"].split("-")[:2])
            # cards: every changed image or prompt (one card per picture: skip Q2, it repeats Q1's
            # picture), and the first 2 rows of each task per split for unchanged tasks
            changed_card = image_changed or o["prompt"] != n["prompt"]
            if changed_card and n["type_of_question"] == "Q2":
                continue
            if not changed_card:
                if shown_per_task.get(task, 0) >= 2:
                    continue
                shown_per_task[task] = shown_per_task.get(task, 0) + 1
            stem = f"{split}_{i:05d}"
            save_thumb(o["image"]["bytes"], os.path.join(out_dir, "img", f"{stem}_v1.png"))
            if image_changed:
                save_thumb(n["image"]["bytes"], os.path.join(out_dir, "img", f"{stem}_v2.png"))
            cards.append({"split": split, "task": task, "changed": changed_card, "new_image": image_changed, "stem": stem,
                          "id": n["ID"], "legacy": o["ID"], "prompt": n["prompt"],
                          "old_prompt": o["prompt"] if o["prompt"] != n["prompt"] else "",
                          "gt": n["ground_truth"], "bias": n["expected_bias"], "meta": n["metadata"],
                          "old_meta": o["metadata"]})
        print(f"{split}: {len(new)} rows", flush=True)
    pd.DataFrame(table).to_csv(os.path.join(out_dir, "rows.csv"), index=False)
    write_html(cards, os.path.join(out_dir, "index.html"))


def write_html(cards, path):
    esc = lambda s: "—" if s is None else html.escape(str(s))
    items = []
    for c in cards:
        imgs = f'<figure><img loading="lazy" src="img/{c["stem"]}_v1.png"><figcaption>v1.0</figcaption></figure>'
        if c["new_image"]:
            imgs += f'<figure><img loading="lazy" src="img/{c["stem"]}_v2.png"><figcaption>v2</figcaption></figure>'
        prompt = esc(c["prompt"])
        if c["old_prompt"]:
            prompt = f'<del>{esc(c["old_prompt"])}</del><br><ins>{prompt}</ins>'
        items.append(f'''<article data-split="{c['split']}" data-task="{c['task']}" data-changed="{int(c['changed'])}">
<div class="imgs">{imgs}</div>
<dl><dt>ID</dt><dd><code>{esc(c['id'])}</code></dd>
<dt>legacy_ID</dt><dd><code>{esc(c['legacy'])}</code></dd>
<dt>split</dt><dd>{c['split']}</dd>
<dt>prompt</dt><dd>{prompt}</dd>
<dt>answer</dt><dd>ground truth <b>{esc(c['gt'])}</b> · expected bias {esc(c['bias'])}</dd>
<dt>metadata</dt><dd><details><summary>v2</summary><pre>{esc(json.dumps(json.loads(c['meta']), indent=1, ensure_ascii=False))}</pre></details>
<details><summary>v1.0</summary><pre>{esc(c['old_meta'])}</pre></details></dd></dl></article>''')
    splits = sorted({c["split"] for c in cards})
    tasks = sorted({c["task"] for c in cards})
    opts = lambda xs: "".join(f'<option value="{x}">{x}</option>' for x in xs)
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>VLMBias v2 review</title>
<style>
:root {{ --bg:#fff; --fg:#1a1a1a; --muted:#666; --card:#f6f6f4; --line:#ddd; --del:#a33; --ins:#1a6b2f; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#161616; --fg:#eee; --muted:#aaa; --card:#222; --line:#333; --del:#e88; --ins:#7c7; }} }}
body {{ background:var(--bg); color:var(--fg); font:14px/1.45 system-ui, sans-serif; margin:0 auto; max-width:1200px; padding:16px; }}
header {{ position:sticky; top:0; background:var(--bg); padding:8px 0; border-bottom:1px solid var(--line); z-index:1; }}
select, label {{ margin-right:12px; }} #count {{ color:var(--muted); }}
article {{ background:var(--card); border-radius:8px; padding:12px; margin:12px 0; display:grid; grid-template-columns:minmax(0,1.2fr) minmax(0,1fr); gap:12px; }}
@media (max-width:800px) {{ article {{ grid-template-columns:1fr; }} }}
.imgs {{ display:flex; gap:8px; flex-wrap:wrap; }} figure {{ margin:0; }} img {{ max-width:100%; background:#fff; border:1px solid var(--line); }}
figcaption {{ color:var(--muted); font-size:12px; }} dl {{ margin:0; display:grid; grid-template-columns:auto 1fr; gap:4px 10px; }}
dt {{ color:var(--muted); }} dd {{ margin:0; overflow-wrap:anywhere; }} del {{ color:var(--del); }} ins {{ color:var(--ins); text-decoration:none; }}
pre {{ white-space:pre-wrap; font-size:12px; margin:4px 0; }}
</style></head><body>
<header><h1 style="margin:0 0 6px;font-size:18px">VLMBias dataset: v1.0 vs v2</h1>
<label>split <select id="split"><option value="">all</option>{opts(splits)}</select></label>
<label>task <select id="task"><option value="">all</option>{opts(tasks)}</select></label>
<label><input type="checkbox" id="onlychanged" checked> changed rows only</label>
<span id="count"></span><br><small>Changed rows (new image or question): one card per picture (Q2 repeats Q1's picture); a card with one image kept its v1 image. Unchanged tasks: 2 example rows per task and split, to check the new IDs and metadata. Every row is in rows.csv.</small></header>
<main>{"".join(items)}</main>
<script>
const f = () => {{
  const s = split.value, t = task.value, oc = onlychanged.checked; let n = 0;
  document.querySelectorAll("article").forEach(a => {{
    const show = (!s || a.dataset.split === s) && (!t || a.dataset.task === t) && (!oc || a.dataset.changed === "1");
    a.style.display = show ? "" : "none"; n += show;
  }});
  count.textContent = n + " shown";
}};
[split, task, onlychanged].forEach(e => e.addEventListener("change", f)); f();
</script></body></html>'''
    open(path, "w").write(page)


if __name__ == "__main__":
    main(*sys.argv[1:4])
