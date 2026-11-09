"""Score answers saved by run_api.py: accuracy and bias rate per task, and their mean.

    python evaluation/score.py results/gpt-4.1_main.jsonl [more.jsonl ...]

The answer is the last {...} in the response, as the prompts request:
  counting questions -> the first integer inside the brackets
  Yes/No questions   -> "Yes" or "No" (case-insensitive)
  other questions    -> the bracket text as-is (identification split)
A row is correct when the answer equals ground_truth, and bias-aligned when it equals
expected_bias (the count or answer of the familiar, unmodified subject). Rows with no
bracketed answer count as wrong and not bias-aligned; rows that failed at the API are
left out (re-run run_api.py to retry them). The mean is over tasks (topics),
as reported in the paper.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict

BRACKET = re.compile(r"\{([^{}]*)\}")


def parse_answer(text: str | None, ground_truth: str) -> str | None:
    found = BRACKET.findall(text or "")
    if not found:
        return None
    raw = found[-1].strip()
    if ground_truth.strip().lower() in ("yes", "no"):
        return raw.capitalize() if raw.lower() in ("yes", "no") else None
    if ground_truth.strip().lstrip("-").isdigit():
        m = re.search(r"-?\d+", raw)
        return m.group() if m else None
    return raw


def same(a: str | None, b) -> bool:
    return a is not None and b is not None and str(a).strip().lower() == str(b).strip().lower()


def score(path: str) -> None:
    latest = {}  # a retried row appears twice; keep its last answered line
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                if rec.get("error") is None or rec["ID"] not in latest:
                    latest[rec["ID"]] = rec
    errors = sum(r.get("error") is not None for r in latest.values())

    by_topic = defaultdict(lambda: [0, 0, 0, 0])  # rows, correct, biased, unparsed
    for r in latest.values():
        if r.get("error") is not None:
            continue  # never answered; re-run run_api.py to fill these in
        ans = parse_answer(r.get("response"), str(r["ground_truth"]))
        t = by_topic[r["topic"]]
        t[0] += 1
        t[1] += same(ans, r["ground_truth"])
        t[2] += same(ans, r.get("expected_bias"))
        t[3] += ans is None

    print(f"\n{path}  ({len(latest) - errors} scored rows)")
    if errors:
        print(f"WARNING: {errors} rows failed at the API and are left out; re-run run_api.py to retry them.")
    if not by_topic:
        return
    print(f"{'Task':<22}{'Rows':>6}{'Accuracy %':>12}{'Bias rate %':>13}{'No answer':>11}")
    accs, biases = [], []
    for topic in sorted(by_topic):
        n, c, b, u = by_topic[topic]
        accs.append(100 * c / n)
        biases.append(100 * b / n)
        print(f"{topic:<22}{n:>6}{accs[-1]:>12.2f}{biases[-1]:>13.2f}{u:>11}")
    print(f"{'Mean over tasks':<22}{'':>6}{sum(accs) / len(accs):>12.2f}{sum(biases) / len(biases):>13.2f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for p in sys.argv[1:]:
        score(p)
