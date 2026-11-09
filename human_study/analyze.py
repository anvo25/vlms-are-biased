"""Reproduce the human-study table and demographics from the paper.

    python human_study/analyze.py
"""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
resp = pd.read_csv(HERE / "responses.csv")
demo = pd.read_csv(HERE / "demographics.csv")


def by_time(g: pd.DataFrame) -> pd.Series:
    cf = g[g.image_type == "counterfactual"]
    og = g[g.image_type == "original"]
    return pd.Series({
        "Participants": g.participant_id.nunique(),
        "Counterfactual acc. (%)": 100 * cf.is_correct.mean(),
        "Original acc. (%)": 100 * og.is_correct.mean(),
        "Overall acc. (%)": 100 * g.is_correct.mean(),
        "Bias-aligned rate (%)": 100 * cf.is_bias_aligned.mean(),
    })


table = resp.groupby("viewing_time_s")[resp.columns].apply(by_time)
table["Participants"] = table["Participants"].astype(int)
print("Human accuracy by image-viewing time (s)\n")
print(table.round(2).to_string(), "\n")

count = demo.set_index(["field", "category"])["count"]
n = int(count["gender"].sum())
age_mid = {"18–24": 21, "25–34": 29.5, "35–44": 39.5}
higher = ["Bachelor's degree", "Master's degree", "Doctorate / PhD / professional degree"]
print(f"Participants: {n}")
print(f"Mean age (age-group midpoints): {sum(age_mid[a] * c for a, c in count['age_group'].items()) / n:.1f}")
print(f"Bachelor's degree or higher: {100 * count['education'][higher].sum() / n:.1f}%")
for g, c in count["gender"].items():
    print(f"{g}: {c} ({100 * c / n:.1f}%)")
