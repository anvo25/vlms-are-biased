"""Publish dataset v2 to the Hugging Face repository (v2 becomes the default config).

Usage: python upload.py <v2 dir> --date YYYY-MM-DD [--repo anvo25/vlms-are-biased] [--dry-run]

Steps (each one is skipped if already done, so the script can be re-run):
  1. tag the commit that is published now as `v1.0`, so the repository as it was stays loadable
     with revision="v1.0"
  2. one commit: add the v2 parquet files under data/v2/, replace the dataset card with
     hf_dataset/README.md (v2 = default config, v1 = the existing files), and delete the three
     v1 files no config uses (data/notitle-*, data/train-*, data/yes_no-*; still in `v1.0`)
  3. tag that commit `v2.0`
The v1 parquet files are not touched, so the `v1` config is the published data.
"""
import argparse
import os
import sys

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

SPLITS = ["main", "identification", "withtitle", "original", "remove_background_q1q2", "remove_background_q3"]
UNUSED_V1 = ("data/notitle-", "data/train-", "data/yes_no-")
V1_SHA = "3761f9fd7163577534a7816c8bc1004035f2e2a0"  # the published v1 commit (2025-12-12)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("v2_dir")
    p.add_argument("--date", required=True, help="release date for the card's news line")
    p.add_argument("--repo", default="anvo25/vlms-are-biased")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    api = HfApi()
    kw = dict(repo_id=a.repo, repo_type="dataset")
    refs = api.list_repo_refs(**kw)
    tags = {t.name: t.target_commit for t in refs.tags}
    head = api.dataset_info(a.repo).sha
    files = api.list_repo_files(**kw)
    print(f"repo {a.repo}: head {head[:8]}, tags {sorted(tags)}")

    # 1. v1.0 tag
    if "v1.0" in tags:
        print(f"[1] tag v1.0 exists ({tags['v1.0'][:8]})")
    else:
        if head != V1_SHA and a.repo == "anvo25/vlms-are-biased":
            sys.exit(f"head is {head}, not the published v1 commit {V1_SHA}; check before tagging")
        print(f"[1] tag v1.0 -> {head[:8]}")
        if not a.dry_run:
            api.create_tag(**kw, tag="v1.0", revision=head,
                           tag_message="v1: the data of the ICLR 2026 paper, as published")

    # 2. v2 commit
    card = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md")).read()
    if "DATE" in card:
        card = card.replace("DATE", a.date)
    ops = [CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card.encode())]
    for s in SPLITS:
        src = os.path.join(a.v2_dir, "data", f"{s}.parquet")
        ops.append(CommitOperationAdd(path_in_repo=f"data/v2/{s}-00000-of-00001.parquet", path_or_fileobj=src))
    ops += [CommitOperationDelete(path_in_repo=f) for f in files if f.startswith(UNUSED_V1)]
    if "v2.0" in tags:
        print(f"[2] tag v2.0 exists ({tags['v2.0'][:8]}); not committing again")
        return
    for op in ops:
        kind = "add   " if isinstance(op, CommitOperationAdd) else "delete"
        size = os.path.getsize(op.path_or_fileobj) / 1e6 if kind == "add   " and isinstance(op.path_or_fileobj, str) else 0
        print(f"[2] {kind} {op.path_in_repo}" + (f" ({size:.0f} MB)" if size else ""))
    if a.dry_run:
        print("dry run: nothing changed")
        return
    info = api.create_commit(**kw, operations=ops, commit_message="v2: red optical-illusion targets, readable IDs, fixes",
                             commit_description="See the dataset card, section 'What changed in v2'. "
                                                "v1 (the ICLR paper's data) stays available as the `v1` config and the `v1.0` tag.")
    print(f"[2] committed {info.oid[:8]}")

    # 3. v2.0 tag
    api.create_tag(**kw, tag="v2.0", revision=info.oid, tag_message="v2: default config")
    print(f"[3] tag v2.0 -> {info.oid[:8]}")


if __name__ == "__main__":
    main()
