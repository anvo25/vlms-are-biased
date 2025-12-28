# Vision Language Models are Biased

<div align="center">    
  <p style="font-size: 20px;">by 
    <a href="https://anvo25.github.io/">An Vo</a><sup>1*</sup>,
    <a href="https://nkn002.github.io/">Khai-Nguyen Nguyen</a><sup>2*</sup>,
    <a href="https://taesiri.ai/">Mohammad Reza Taesiri</a><sup>3</sup>, 
    <a href="https://www.linkedin.com/in/dang-thi-tuong-vy-00a357278/">Vy Tuong Dang</a><sup>1</sup>, 
    <a href="https://anhnguyen.me/research/">Anh Totti Nguyen</a><sup>4†</sup>, 
    <a href="https://www.resl.kaist.ac.kr/members/director">Daeyoung Kim</a><sup>1†</sup>
  </p>
  <p>
    <sup>*</sup>Equal contribution<br>
    <sup>†</sup>Equal advising<br>
    <sup>1</sup>KAIST, <sup>2</sup>College of William and Mary, <sup>3</sup>University of Alberta, <sup>4</sup>Auburn University
  </p>

[![Project Page](https://img.shields.io/badge/Project_Page-vlmsarebiased.github.io-blue.svg)](https://vlmsarebiased.github.io)
[![arXiv](https://img.shields.io/badge/arXiv-2505.23941-b31b1b.svg)](https://arxiv.org/abs/2505.23941)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-yellow.svg)](https://huggingface.co/datasets/anvo25/vlms-are-biased)
[![Code License](https://img.shields.io/badge/Code_License-MIT-green.svg)](LICENSE)

</div>

---

## 📌 Abstract

<p align="center">
  <!-- Insert key figure from your paper here -->
  <img src="./figures/overview.png" alt="VLMBias Overview" width="80%"/>
</p>

*Large language models (LLMs) memorize a vast amount of prior knowledge from the Internet that help them on downstream tasks but also may notoriously sway their outputs towards wrong or biased answers. In this work, we test how the knowledge about popular subjects hurt the accuracy of vision language models (VLMs) on standard, objective visual tasks of counting and identification. We find that state-of-the-art VLMs are **strongly biased** (e.g., unable to recognize a fourth stripe has been added to a 3-stripe Adidas logo) scoring an average of 17.05% accuracy in counting (e.g., counting stripes in an Adidas-like logo) across 7 diverse domains from animals, logos, chess, boardgames, optical illusions, to patterned grids. Insert text (e.g., "Adidas") describing the subject name into the counterfactual image further decreases VLM accuracy. The biases in VLMs are so strong that instructing them to double-check their results or rely exclusively on image details to answer improves counting accuracy by only +2 points, on average. Our work presents an interesting failure mode in VLMs and an automated framework for testing VLM biases. Code and data are available at: [vlmsarebiased.github.io](https://vlmsarebiased.github.io)*

---

## 2. Quick Start

Get started with VLMBias in minutes—choose the option that fits your use case.

### 2.1 Try Example Images

Want to quickly test your model on challenging images? Use our pre-selected examples where most VLMs fail:

👉 **[Download example images](./examples)**

### 2.2 Run with lmms-eval (For Standardized Evaluation)

VLMBias is officially integrated with [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval), a popular evaluation framework.

**Step 1:** Install lmms-eval following their [documentation](https://github.com/EvolvingLMMs-Lab/lmms-eval)

**Step 2:** Run the benchmark:

```bash
python -m lmms_eval \
  --model qwen2_5_vl \
  --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct \
  --tasks vlms_are_biased \
  --batch_size 1 \
  --device cuda:0
```

For more details: [lmms_eval/tasks/vlms_are_biased](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/tasks/vlms_are_biased)

**Note:** lmms-eval currently only supports the `main` subset. For other subsets, use [Section 2.3](#23-use-the-pre-built-dataset-recommended).

### 2.3 Use the Pre-built Dataset (For Thorough Evaluation)

Want to evaluate your model on the full benchmark, or to have more control over the evaluation process? Load our dataset directly from Hugging Face:

```python
import datasets
dataset = datasets.load_dataset('anvo25/vlms-are-biased')
```

See [Section 3](#3-dataset-details) for details on available subsets.


---

## 3. Dataset Details

### 3.1 Dataset Structure

Loading the dataset returns a `DatasetDict` with multiple subsets:

```python
DatasetDict({
    main: Dataset({...num_rows: 2784}),
    identification: Dataset({...num_rows: 1392}),
    withtitle: Dataset({...num_rows: 2784}),
    original: Dataset({...num_rows: 458}),
    remove_background_q1q2: Dataset({...num_rows: 2784}),
    remove_background_q3: Dataset({...num_rows: 1392}),
})
```

### 3.2 Subset Descriptions

| Subset | Description | Paper Section |
|--------|-------------|---------------|
| `main` | Counting dataset of counterfactual images | Throughout paper |
| `identification` | Identification dataset of counterfactual images | Section 4.3 |
| `withtitle` | Counting dataset with in-image title injection | Section A.9 |
| `original` | Identification dataset of original (unmodified) images | Section 4.1 |
| `remove_background_q1q2` | Counting dataset with background removed | Section 4.4 |
| `remove_background_q3` | Identification dataset with background removed | Section 4.4 |

### 3.3 Data Fields

Each sample contains: `image`, `ID`, `image_path`, `topic`, `sub_topic`, `prompt`, `ground_truth`, `expected_bias`, `with_title`, `type_of_question`, `pixel`, `metadata`

---

## 4. Generate Your Own Dataset

**Goal:** Reproduce our dataset or create custom variations. Skip this section if you only need the pre-built dataset.

### 4.1 Installation

```bash
git clone https://github.com/anvo25/vlms-are-biased.git
cd vlms-are-biased
pip install -r requirements.txt
```

### 4.2 Generate All Datasets

```bash
# Step 1: Generate images without titles
python main.py --all

# Step 2: Add in-image titles
python add_titles.py --topic all
```

### 4.3 Generate Specific Tasks

To run the code to generate the **counterfactual** images for a specific task, go to the following embedded links:
- **Chess Pieces**: [Chess pieces](generators/chess_pieces_generator.py), [Xiangqi pieces](generators/xiangqi_board_generator.py) (modified starting positions)
- **Game Boards**: [Chess board](generators/chess_board_generator.py), [Go board](generators/go_board_generator.py), [Xiangqi board](generators/xiangqi_board_generator.py), [Sudoku board](generators/sudoku_board_generator.py) (dimension variations)  
- **[Optical Illusions](generators/optical_illusion_generator.py)**: Ebbinghaus, Müller-Lyer, Ponzo, Vertical-Horizontal, Zöllner, Poggendorff  
- **[Patterned Grids](generators/patterned_grid_generator.py)**: Dice patterns, Tally mark patterns (anomalous cells)  
- **[Animals](generators/animals_generator.py)**: Mammals (4 legs → 5 legs) and birds (2 legs → 3 legs)  
- **[Logos](generators/logos_generator.py#L10)**: Consisting of 2 logo types: shoes and cars.
  - **Shoe Logos**:  
    - Nike (1 swoosh → 2 swooshes)
    - Adidas (3 stripes → 4 stripes)
  - **Car Logos**:  
    - Maserati (3 prongs → 5 prongs)
    - Mercedes-Benz (3-pointed star → 4-pointed star)
    - Audi (4 overlapping circles → 5 overlapping circles)
- **[Flags](generators/flags_generator.py)**: Star-typed flags (+1 and −1 star) and stripes (+1 and −1 stripe)

*All images are generated at 384px, 768px, and 1152px resolutions.*

### 4.4 Adding Titles to Generated Images

After generating images, add in-image titles:

```bash
python add_titles.py --topic chess_pieces  # or: all, logos, flags, etc.
```

---

## 5. Project Structure

```
vlms-are-biased/
├── main.py                          # Generate "notitle" datasets
├── add_titles.py                    # Add "in_image_title" versions
├── generators/                      # Individual dataset generators
│   ├── chess_pieces_generator.py
│   ├── optical_illusion_generator.py
│   └── ...
├── examples/                        # Example images for quick testing
├── vlms-are-biased-notitle/         # Output: images without titles
└── vlms-are-biased-in_image_title/  # Output: images with titles
```

---

## 6. Citation

```bibtex
@misc{vlmsarebiased,
      title={Vision Language Models are Biased}, 
      author={An Vo and Khai-Nguyen Nguyen and Mohammad Reza Taesiri and Vy Tuong Dang and Anh Totti Nguyen and Daeyoung Kim},
      year={2025},
      eprint={2505.23941},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.23941}, 
}
```


