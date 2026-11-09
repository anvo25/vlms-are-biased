# Human study

Participant-level data for the human study in the paper (Animals task, Table "Human accuracy increases with image-viewing time").

**Task.** Participants counted the legs of an animal. Each person was randomly assigned one image-viewing time (0.2, 0.5, 1.0 or 2.0 s) and answered 10 randomly selected questions: 5 counterfactual images (one leg added) and 5 original images. Reading the question and choosing the answer were not timed. The study ran on the project website on 30 July 2025.

Reproduce the paper's numbers:

```bash
python human_study/analyze.py
```

| Viewing time (s) | Participants | Counterfactual acc. (%) | Original acc. (%) | Overall acc. (%) | Bias-aligned rate (%) |
|---:|---:|---:|---:|---:|---:|
| 0.2 | 18 | 50.00 | 93.33 | 71.67 | 34.44 |
| 0.5 | 16 | 70.00 | 98.75 | 84.38 | 17.50 |
| 1.0 | 28 | 82.86 | 97.86 | 90.36 | 14.29 |
| 2.0 | 16 | 93.75 | 100.00 | 96.88 | 5.00 |

## Files

| File | Contents |
|---|---|
| `participants.csv` | One row per participant (78): viewing time and scores |
| `responses.csv` | One row per answered question (780) |
| `demographics.csv` | Aggregate counts of gender, age group and education |
| `VLMBias_human_evaluation.xlsx` | All of the above, plus a `Summary` sheet that recomputes the table with formulas |
| `analyze.py` | Prints the table and demographics from the CSV files |

### `participants.csv`

| Column | Description |
|---|---|
| `participant_id` | Anonymous ID (P01–P78), assigned in random order |
| `viewing_time_s` | Assigned image-viewing time in seconds |
| `n_questions`, `n_correct`, `n_counterfactual_correct`, `n_original_correct` | Counts over the 10 questions |

### `responses.csv`

| Column | Description |
|---|---|
| `participant_id`, `viewing_time_s` | As above |
| `trial` | Question position for this participant (1–10) |
| `image_id` | Image name in the VLMBias Animals set (prompt Q1, no title, resolution suffix) |
| `animal_id`, `animal` | Animal index (1–91) and name |
| `image_type` | `counterfactual` (one leg added) or `original` |
| `resolution_px` | Image resolution shown (384, 768 or 1152) |
| `question` | Question text as shown. The first 30 participants saw "How many legs *do* this animal have?" (a typo, fixed during the study) |
| `correct_answer` | Leg count in the image |
| `bias_answer` | Leg count of the unmodified animal (counterfactual images only) |
| `user_answer` | Participant's answer (multiple choice) |
| `is_correct` | 1 if `user_answer` equals `correct_answer` |
| `is_bias_aligned` | 1 if `user_answer` equals `bias_answer` (counterfactual images only) |
| `trial_time_ms` | Time from question onset to answer, in ms (reading + image + response) |

## Exclusions and privacy

The raw export had 82 submissions. We removed 3 internal test submissions and 1 submission whose per-question responses were not recorded, leaving the 78 participants reported in the paper.

Participation was anonymous. Before release we removed the optional email field, the original submission IDs and all wall-clock timestamps, assigned participant IDs in random order, and kept demographics only as aggregate counts, so that no row can be linked to a person.
