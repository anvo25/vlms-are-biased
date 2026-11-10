"""New row IDs for dataset v2 (the v1 ID is kept in the `legacy_ID` column).

Format: task-subtask-NNN-detail-condition-question-px, e.g.

    illusion-ebbinghaus-013-diff0p5-notitle-q1-384
    pieces-chess-001-remove_g1-nobg-q3-768
    logos-car-004-audi_black_coupe-original-count-na

Fields are separated by "-" and words inside a field by "_", so `ID.split("-")` always gives
the 7 fields:

    task, subtask   what is tested (animals-legs, logos-car, boards-go, illusion-ponzo, ...)
    NNN             sample number within task-subtask and condition
    detail          what the sample is: the animal, brand, flag, or the modification
    condition       notitle (main, identification), withtitle, nobg (remove_background_*),
                    original (unmodified images)
    question        q1, q2 (counting), q3 (identification); original split: count, id
    px              image height in pixels (384, 768, 1152); "na" for original photos

Sample numbers are shared by notitle, withtitle and identification (the same images). nobg
images were generated separately for pieces, boards, grids and illusions, so a nobg sample
with the same number can show a different modification; `detail` tells them apart.
"""
import json
import re

SPLIT_CONDITION = {
    "main": "notitle", "identification": "notitle", "withtitle": "withtitle",
    "remove_background_q1q2": "nobg", "remove_background_q3": "nobg", "original": "original",
}

SUBTASK = {
    "Animal Add Legs": "animals-legs", "animal": "animals-legs",
    "Car Logos": "logos-car", "cars": "logos-car",
    "Shoe Logos": "logos-shoe", "shoes": "logos-shoe",
    "Flags 2D Stars": "flags-stars", "Flags 2D Stripes": "flags-stripes", "flags": "flags-{kind}",
    "Chess Pieces": "pieces-chess", "Xiangqi Pieces": "pieces-xiangqi",
    "Chess Grid": "boards-chess", "Chess Board": "boards-chess",
    "Go Grid": "boards-go", "Go Board": "boards-go",
    "Sudoku Grid": "boards-sudoku", "Sudoku Board": "boards-sudoku",
    "Xiangqi Grid": "boards-xiangqi", "Xiangqi Board": "boards-xiangqi",
    "Dice Pattern": "grid-dice", "Tally Pattern": "grid-tally",
}
ILLUSION_SLUG = {"Ebbinghaus": "ebbinghaus", "MullerLyer": "mullerlyer", "Ponzo": "ponzo",
                 "VerticalHorizontal": "verticalhorizontal", "Zollner": "zollner",
                 "Poggendorff": "poggendorff"}


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def basename(row):
    return row["image_path"].split("/")[-1].rsplit(".", 1)[0]


def metadata(row):
    md = row["metadata"]
    while isinstance(md, str):  # may be empty, JSON, or JSON-encoded JSON (original split)
        md = json.loads(md) if md.strip() else {}
    return md or {}


def subtask(row, flag_kind=None):
    st = row["sub_topic"]
    if "llusion" in row["topic"]:
        return "illusion-" + ILLUSION_SLUG[basename(row).split("_")[0]]
    s = SUBTASK[st]
    if "{kind}" in s:  # original flags: stars or stripes, from that flag's counting question
        s = s.format(kind=flag_kind[basename(row)])
    return s


def question(row):
    q = row["type_of_question"]
    if row["split"] == "original":
        return {"ID": "id", "Counting/Illusion": "count"}[q]
    return q.lower()


def px(row):
    if int(row["pixel"]) > 0:
        return str(row["pixel"])
    m = re.search(r"_px(\d+)", row["ID"])  # original split: pixel is 0, the v1 ID may say it
    return m.group(1) if m else "na"


def number_and_detail(row, st):
    """(sample number or None, detail) from the v1 ID, image file name and metadata."""
    lid, base, md = row["ID"], basename(row), metadata(row)
    m = re.match(r"[A-Za-z_]+?_(\d+)_", lid)
    num = int(m.group(1)) if m else None

    if row["split"] == "original":
        if st.startswith("illusion"):
            return num, "diff0"
        if st.startswith(("boards", "pieces")):
            return 1, "standard"
        if st.startswith("logos"):
            return None, slug(re.sub(r"_\d+$", "", base))
        subject = re.sub(r"_prompt\d+$", "", lid)
        return None, slug(re.sub(r"^flag_of_(the_)?", "", subject))

    if st == "animals-legs":
        return num, slug(re.match(r"animal_\d+_(.+?)_(?:notitle|withtitle)", lid).group(1))
    if st.startswith("logos"):
        return num, slug(base.split("-")[0])
    if st.startswith("flags"):
        m = re.match(r"(?i)flag[ _]of[ _](?:the[ _])?(.+?)-(?:stars|stripes)=(\d+)", base)
        return (num if st == "flags-stars" else None), f"{slug(m.group(1))}_{m.group(2)}"
    if st.startswith("pieces"):
        action = re.search(r"_(remove|replace)_", lid).group(1)
        square = re.search(r"_at_([a-z]\d+)", base).group(1)
        return num, f"{action}_{square}"
    if st.startswith("grid"):
        m = re.search(r"_(remove|replace|add)_([A-Z]\d+)", base)
        return num, f"{m.group(1)}_{m.group(2).lower()}"
    if st.startswith("boards"):
        m = re.match(r"[a-z]+_(?:grid|board)_\d+_(.+?)(?:_notitle|_withtitle)?_px\d+", base)
        detail = m.group(1)
        if st == "boards-go" and row["split"] in ("main", "identification", "withtitle"):
            # e.g. "remove_row" with remove_index 18 on a 19x19 board -> "remove_row_last"
            idx = md.get("remove_index") if "remove" in detail else md.get("insert_index")
            detail += "_first" if idx == 0 else "_last"
        detail = re.sub(r"^(row|col)_", "", detail)
        return num, slug(detail)
    if st.startswith("illusion"):
        return num, re.search(r"_(diff(?:neg)?[0-9p]+)", base).group(1)
    raise ValueError(f"no ID rule for {st}: {lid}")


def assign_ids(rows):
    """Add 'ID' (new) and 'legacy_ID' to each row dict. Rows need split, ID, image_path,
    topic, sub_topic, prompt, type_of_question, pixel, metadata."""
    flag_kind = {basename(r): ("stars" if "star" in r["prompt"].lower() else "stripes")
                 for r in rows if r["sub_topic"] == "flags" and r["type_of_question"] != "ID"}
    parsed = []
    for r in rows:
        st = subtask(r, flag_kind)
        num, detail = number_and_detail(r, st)
        parsed.append((st, SPLIT_CONDITION[r["split"]], num, detail))
    # Samples without a number in v1 (flag stripes, original photos) are numbered by detail,
    # in the same order for every condition.
    missing = {}
    for (st, cond, num, detail) in parsed:
        if num is None:
            missing.setdefault((st, cond), set()).add(detail)
    order = {k: {d: i + 1 for i, d in enumerate(sorted(v))} for k, v in missing.items()}
    for r, (st, cond, num, detail) in zip(rows, parsed):
        if num is None:
            num = order[(st, cond)][detail]
        r["legacy_ID"] = r["ID"]
        r["ID"] = "-".join([st, f"{num:03d}", detail, cond, question(r), px(r)])
    return rows
