# How to use `train.py`

## Prerequisites
- Python 3.9+
- Dependencies: scikit-learn, pandas, tqdm
- Data at `data/train.jsonl` and `data/test.jsonl` (or use `--kaggle`)

## Basic run
```zsh
python train.py
```

## Arguments
- `--show`: print the structure of one train battle.
- `--idx INT`: battle index to inspect with `--show` (default 0).
- `--n_turns INT`: number of turns to show with `--show` (default 2).
- `--show_train`: print the structure of one test battle.
- `--kaggle`: use Kaggle competition paths.

## Examples
- Inspect first 2 turns of battle 0 and train:
```zsh
python train.py --show --idx 0 --n_turns 2
```

- Run on Kaggle:
```zsh
python train.py --kaggle
```

## Output
- Logs with Train Accuracy and Validation Accuracy.
- A `submission.csv` file created in the project root.