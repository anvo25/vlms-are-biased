# Vision Language Models are Biased

## Generated Datasets

VLMBias can generate various types of visual test datasets:

- **Game Pieces**
  - Chess pieces
  - Xiangqi (Chinese chess) pieces

- **Game Boards**
  - Chess grid
  - Go grid
  - Xiangqi grid
  - Sudoku grid

- **Counting and Recognition**
  - Dice
  - Tally marks

- **Perception Tests**
  - Optical illusions

## How to Run

```bash
# Generate all datasets
python vlmbias.py --all

# Generate specific datasets
python vlmbias.py --chess_pieces
python vlmbias.py --xiangqi_pieces
python vlmbias.py --chess_grid
python vlmbias.py --go_grid
python vlmbias.py --xiangqi_grid
python vlmbias.py --sudoku_grid
python vlmbias.py --dice_tally
python vlmbias.py --illusion



