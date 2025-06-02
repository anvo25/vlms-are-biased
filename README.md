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

## 💻 Getting Started

```bash
git clone https://github.com/anvo25/vlms-are-biased.git
cd VLMBias
pip install -r requirements.txt
```

---

## 🚀 Quick Example

Generate chess pieces dataset with modified starting positions:
```bash
# Step 1: Generate "notitle" images
python main.py --chess_pieces

# Step 2: Add titles to create "in_image_title" versions  
python add_titles.py --topic chess_pieces
```

Generate all available datasets:
```bash
python main.py --all
python add_titles.py --topic all
```

Generate specific optical illusions:
```bash
python main.py --optical_illusions --illusion_type Ebbinghaus
```

## 📊 Tasks

1. To download the images in our benchmark for a specific task, go to our [huggingface dataset](https://huggingface.co/datasets/anvo25/vlms-are-biased) and select the task topic.

2. To run the code to generate the images for a specific task, go to the following embedded links:
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




   
---

## 📂 Structure

```
vlms-are-biased/
├── main.py                        # Generate "notitle" datasets
├── add_titles.py                  # Add "in_image_title" versions
├── generators/                    # Individual dataset generators
│   ├── chess_pieces_generator.py
│   ├── optical_illusion_generator.py
│   └── ...
├── vlms-are-biased-notitle/       # Output: images without titles
└── vlms-are-biased-in_image_title/ # Output: images with titles
```

---

## 📖 Citation

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