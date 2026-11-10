---
language:
- en
license: mit
size_categories:
- 10K<n<100K
task_categories:
- visual-question-answering
- image-text-to-text
pretty_name: Vision Language Models are Biased
configs:
- config_name: default
  data_files:
  - split: main
    path: data/v2/main-*
  - split: identification
    path: data/v2/identification-*
  - split: withtitle
    path: data/v2/withtitle-*
  - split: original
    path: data/v2/original-*
  - split: remove_background_q1q2
    path: data/v2/remove_background_q1q2-*
  - split: remove_background_q3
    path: data/v2/remove_background_q3-*
- config_name: v1
  data_files:
  - split: main
    path: data/main-*
  - split: identification
    path: data/identification-*
  - split: withtitle
    path: data/withtitle-*
  - split: original
    path: data/original-*
  - split: remove_background_q1q2
    path: data/remove_background_q1q2-*
  - split: remove_background_q3
    path: data/remove_background_q3-*
---

# Vision Language Models are Biased

<div align="center">
  <p style="font-size: 20px;">by
    <a href="https://anvo25.github.io/">An Vo</a><sup>1*</sup>,
    <a href="https://nkn002.github.io/">Khai-Nguyen Nguyen</a><sup>2*</sup>,
    <a href="https://taesiri.ai/">Mohammad Reza Taesiri</a><sup>3</sup>, <br>
    <a href="https://www.linkedin.com/in/dang-thi-tuong-vy-00a357278/">Vy Tuong Dang</a><sup>1</sup>,
    <a href="https://anhnguyen.me/research/">Anh Totti Nguyen</a><sup>4†</sup>,
    <a href="https://www.resl.kaist.ac.kr/members/director">Daeyoung Kim</a><sup>1†</sup>
  </p>
  <p>
    <sup>*</sup>Equal contribution    <sup>†</sup>Equal advising<br>
    <sup>1</sup>KAIST, <sup>2</sup>College of William and Mary, <sup>3</sup>University of Alberta, <sup>4</sup>Auburn University
  </p>
  <h3>ICLR 2026</h3>

[![Website](https://img.shields.io/badge/Project_Page-vlmsarebiased.github.io-blueviolet)](https://vlmsarebiased.github.io)
[![arXiv](https://img.shields.io/badge/arXiv-2505.23941-b31b1b.svg)](https://arxiv.org/abs/2505.23941)
[![Code](https://img.shields.io/badge/Code-GitHub-black)](https://github.com/anvo25/vlms-are-biased)

</div>

> **News (v2, 2026-11-10).** The default version of this dataset is now **v2** (the `default` config). In v2 the optical-illusion
> features that the questions ask about are drawn in red and the questions name them; the other
> images and every answer are unchanged. Rows also have new, readable IDs. The data of the ICLR paper
> is still here: load the `v1` config, or the repository exactly as published with `revision="v1.0"`.
> See [What changed in v2](#what-changed-in-v2).

**TL;DR:** State-of-the-art vision language models (VLMs) count perfectly on original images but
fail when a well-known subject is subtly modified (e.g., a 4th stripe added to the Adidas logo),
because they answer from memorized knowledge instead of the image.

## Abstract

*Large language models (LLMs) memorize a vast amount of prior knowledge from the Internet that helps them on downstream tasks but also may notoriously sway their outputs toward wrong or biased answers. In this work, we test how the knowledge of popular subjects hurts the accuracy of vision language models (VLMs) on standard, objective visual tasks of counting and identification. We find that state-of-the-art VLMs are **strongly biased** (e.g., unable to recognize that a 4th stripe has been added to a 3-stripe Adidas logo), scoring an average of 17.05% accuracy in counting (e.g., counting stripes in an Adidas-like logo) across 7 diverse domains spanning animals, logos, chess, game boards, optical illusions, and patterned grids. Removing image backgrounds nearly doubles accuracy (by 21.09 points), revealing that background visual cues trigger these biased responses. Further analysis of VLMs' reasoning patterns shows that counting accuracy initially rises with thinking tokens, reaching ~40%, before declining with model overthinking. Our work presents an interesting failure mode in VLMs and a human-supervised automated framework for testing VLM biases.*

## Load the data

```python
from datasets import load_dataset

# v2 (the `default` config)
main = load_dataset("anvo25/vlms-are-biased", split="main")

# v1: the data of the ICLR paper
main_v1 = load_dataset("anvo25/vlms-are-biased", "v1", split="main")

# the repository exactly as it was published (v1 files and dataset card)
main_v1 = load_dataset("anvo25/vlms-are-biased", split="main", revision="v1.0")
```

| Split | Rows | Contents | Paper |
|---|---:|---|---|
| `main` | 2,784 | Counting questions (Q1, Q2) on counterfactual images | Throughout |
| `identification` | 1,392 | Identification questions (Q3) on the same images | Sec. 4.3 |
| `withtitle` | 2,784 | `main` with the subject's name written in the image | Sec. A.9 |
| `original` | 458 | Questions on the original (unmodified) images | Sec. 4.1 |
| `remove_background_q1q2` | 2,784 | Counting, with the image background removed | Sec. 4.4 |
| `remove_background_q3` | 1,392 | Identification, with the image background removed | Sec. 4.4 |

The 7 tasks are animals, logos, flags, chess pieces, game boards, optical illusions and patterned grids.

### Columns (v2)

| Column | Description |
|---|---|
| `image` | The image (PNG) |
| `ID` | Row ID, e.g. `illusion-ponzo-013-diff0p15-notitle-q1-384` (format below) |
| `legacy_ID` | The row's ID in v1, to match results computed on v1 |
| `topic`, `sub_topic` | Task and subtask, e.g. `Optical Illusion` / `Ponzo illusion` |
| `prompt` | The question, as sent to the model. It asks for the answer in curly brackets |
| `ground_truth` | The correct answer |
| `expected_bias` | The biased answer: the answer for the unmodified subject (empty in `original`) |
| `with_title` | Whether the subject's name is written in the image |
| `type_of_question` | `Q1`, `Q2` (counting), `Q3` (identification); in `original`: `Counting/Illusion`, `ID` |
| `pixel` | Image height in pixels: 384, 768 or 1152 (0 in `original`) |
| `metadata` | JSON string with the settings of the row's task (e.g., the modified square, the illusion parameters) and `source_image`, the original file name |

**ID format:** `task-subtask-NNN-detail-condition-question-px`, so `ID.split("-")` gives 7 fields:

| Field | Values |
|---|---|
| task-subtask | `animals-legs`, `logos-car`, `logos-shoe`, `flags-stars`, `flags-stripes`, `pieces-chess`, `pieces-xiangqi`, `boards-chess`, `boards-go`, `boards-sudoku`, `boards-xiangqi`, `grid-dice`, `grid-tally`, `illusion-ebbinghaus`, `illusion-mullerlyer`, `illusion-ponzo`, `illusion-verticalhorizontal`, `illusion-zollner`, `illusion-poggendorff` |
| NNN | Sample number. `main`, `identification` and `withtitle` share numbers (the same images) |
| detail | What the sample is: the animal (`horse`), brand (`audi`), flag and count (`uruguay_8`), or the modification (`remove_g1`, `add_first_row`, `diff0p3`) |
| condition | `notitle` (`main`, `identification`), `withtitle`, `nobg` (`remove_background_*`), `original` |
| question | `q1`, `q2`, `q3`; in `original`: `count`, `id` |
| px | 384, 768, 1152, or `na` |

For pieces, boards, grids and illusions, the `nobg` images were generated separately from `main`,
so a `nobg` sample with the same number can show a different modification; `detail` tells them apart.

## Evaluation frameworks

| Framework | Task / dataset name | Data it uses |
|---|---|---|
| [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/tasks/vlms_are_biased) | `vlms_are_biased` | This repository, default config, `main` split: **v2** |
| [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) | `vlms_are_biased_main` | Its own copy of the v1 `main` split |
| [Our scripts](https://github.com/anvo25/vlms-are-biased#2-evaluate-an-api-model) | `evaluation/run_api.py` | Any split, v2 or v1 (`--config v1`) |

**lmms-eval**

```bash
python -m lmms_eval \
  --model qwen2_5_vl \
  --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct \
  --tasks vlms_are_biased \
  --batch_size 1 \
  --device cuda:0
```

lmms-eval loads this repository's default config, so it runs on v2. To reproduce numbers computed on
v1, copy `lmms_eval/tasks/vlms_are_biased/vlms_are_biased.yaml`, change `task:` to a new name, and add
`dataset_name: v1`.

**VLMEvalKit**

```bash
python run.py --data vlms_are_biased_main --model Qwen2.5-VL-7B-Instruct
```

VLMEvalKit evaluates a copy of the v1 `main` split that it hosts itself, so its results match the
ICLR paper's data and not v2.

**API models (OpenAI, Anthropic, Gemini)**

```bash
git clone https://github.com/anvo25/vlms-are-biased && cd vlms-are-biased
pip install -r requirements.txt
python evaluation/run_api.py --provider openai --model gpt-4.1 --split main            # v2
python evaluation/run_api.py --provider openai --model gpt-4.1 --split main --config v1
python evaluation/score.py results/gpt-4.1_main.jsonl
```

**Scoring.** The answer is the last `{...}` in the response. A row is correct when the answer equals
`ground_truth` and bias-aligned when it equals `expected_bias`. The paper reports the mean over the
7 tasks.

## What changed in v2

The rows, their order, `ground_truth`, `expected_bias`, `type_of_question` and `pixel` are the same as
in v1, and so is every image outside the optical-illusion task (one broken image aside, see below).

**Optical illusions (3,300 rows in all splits).**
- The features the questions ask about are drawn in red (e.g., the two lines to compare, with the
  arrowheads, converging lines and hatching in black). Each image is the v1 picture with only the
  color changed: Ebbinghaus, Müller-Lyer, Ponzo, Zöllner and Poggendorff were redrawn with
  [Pyllusion](https://github.com/RealityBending/Pyllusion) with the parameters that reproduce the v1
  picture pixel for pixel, and Vertical-Horizontal and the `remove_background` images were recolored.
  In `withtitle`, the title is unchanged.
- The counting questions name the red features, e.g., *"Are the two horizontal lines equal in length?"*
  became *"Are the two red lines equal in length?"*. In `remove_background_*`, Poggendorff keeps its
  red and blue segments (*"Are the red and blue line segments aligned?"*), because without the
  occluding bar the two segments touch.
- `metadata.pyllusion_params` gives the exact drawing parameters. The `strength` stored in v1 has the
  opposite sign for most images.

**IDs.** Every row has a new ID in the format above; the v1 ID is in `legacy_ID`. In v1, 246 IDs were
shared by rows of different splits, and in `original` 90 car-logo rows shared 6 IDs (e.g.,
`audi_prompt1` for 15 different photos). v2 IDs are unique across the whole dataset.

**Fixes.**
- `identification`: the image of `shoe_005_notitle_Q3_px768` was a truncated PNG (the first 327,769
  bytes of the file), so decoding the split failed. v2 has the full image, the same as in `main`.
- `remove_background_*`: animals, logos and flags had empty `metadata` and `with_title`; they now
  have the metadata of the same image in `main`, and `with_title` is `false`.
- `topic` and `sub_topic` use the same names in every split (e.g., `remove_background_*` used
  "Optical Illusions" / "Chess Board"). `main`'s names are unchanged.
- `metadata` holds only the settings of the row's own task, with the same key names in every split
  (v1 stored the same 38 keys for every task, most of them empty).
- `image_path` was removed (it was a path on the authors' machine); the file name is in
  `metadata.source_image`.

**Known issue (v1 and v2).** In chess boards, adding or removing the first row or column gives the
same image as adding or removing the last one, and so do the xiangqi-board column edits (6 of the
24 board samples). The answers are correct; the images are duplicates.

## Citation

```bibtex
@inproceedings{vlmsarebiased,
      title={Vision Language Models are Biased},
      author={An Vo and Khai-Nguyen Nguyen and Mohammad Reza Taesiri and Vy Tuong Dang and Anh Totti Nguyen and Daeyoung Kim},
      booktitle={The Fourteenth International Conference on Learning Representations (ICLR)},
      year={2026},
      url={https://arxiv.org/abs/2505.23941},
}
```
