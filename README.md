# A Modular Fusion Neural Network to Predict Multi-Metal Binding Sites in Protein Sequences

This repository contains the reference implementation for the paper **"A Modular Fusion Neural Network Approach to Efficiently Predict Multi-Metal Binding Sites in Protein Sequences"**.

The code trains a residue-level neural network for predicting protein metal-binding sites. It uses separate convolutional models for individual metal ions and a fusion model that combines their predictions into final multi-metal binding-site probabilities.

## Overview

The current implementation supports three metal classes:

- `Zn`
- `Fe`
- `Mg`

The training pipeline in `main.py` performs the following steps:

1. Loads protein sequences and annotated binding sites from `metal_new.csv`.
2. Tokenizes amino-acid sequences at the character level.
3. Builds residue-level labels for each supported metal type.
4. Splits the dataset into training/validation and held-out test sets.
5. Trains one single-metal CNN model per metal type.
6. Trains a fusion model on top of the single-metal model predictions.
7. Evaluates the ensemble across multiple decision thresholds.
8. Saves the best fold's trained models and tokenizer.
9. Runs an example prediction on a protein sequence.

## Repository Structure

```text
.
+-- main.py       # Training, evaluation, model saving, and example prediction
+-- README.md     # Project documentation
+-- LICENSE       # MIT License
```

## Requirements

The script expects Python 3 and the following Python packages:

- `tensorflow`
- `pandas`
- `numpy`
- `scikit-learn`

Install the dependencies with:

```bash
pip install tensorflow pandas numpy scikit-learn
```

For GPU acceleration, install a TensorFlow build compatible with your CUDA/cuDNN environment.

## Dataset

Before running the script, place a CSV file named `metal_new.csv` in the repository root.

The file must contain these columns:

| Column | Description | Example |
| --- | --- | --- |
| `sequence` | Protein sequence using one-letter amino-acid codes | `MTEITAAMVKELRESTGAGMMDCKNALSETQ` |
| `site` | List of 1-based residue positions for known binding sites | `[5, 12, 19]` |
| `metal` | List of metal labels aligned with `site` | `['Zn', 'Fe', 'Zn']` |

Important notes:

- Residue positions are treated as **1-based** in the input file.
- Only positions below `MAX_LEN` are used.
- The default maximum sequence length is `500` residues.
- The `site` and `metal` lists must have matching order and length.
- The default code only labels `Zn`, `Fe`, and `Mg`; other metals are ignored unless `metal_types` is updated.

## Usage

Run the complete training and evaluation pipeline:

```bash
python main.py
```

The script will print:

- Five-fold cross-validation precision, recall, and F1 score.
- Average cross-validation metrics.
- Test-set metrics for multiple thresholds.
- Per-metal precision, recall, and F1 scores.
- Macro and micro averaged scores.
- MCC, AUC, and AUPRC values.
- Example predicted binding sites and probabilities.

## Generated Files

After a successful run, the script writes the following files:

```text
tokenizer.pkl
best_metal_model_Zn.h5
best_metal_model_Fe.h5
best_metal_model_Mg.h5
best_fusion_model.h5
```

These files are generated artifacts and are not required in the source repository unless you want to distribute a trained model.

## Model Configuration

Key settings are defined near the top of `main.py`:

```python
metal_types = ['Zn', 'Fe', 'Mg']
MAX_LEN = 500
N_FOLDS = 5
```

To support additional metals, update `metal_types` and ensure the dataset uses matching labels.

To use longer protein sequences, increase `MAX_LEN`. This also increases memory usage because labels and predictions are stored at residue level.

## Prediction

The script includes an example prediction function:

```python
predict_binding_sites_ensemble(seq, threshold=0.5, top_k=3)
```

It returns a list of predicted binding residues. Each result contains:

- The 1-based residue position.
- The top metal predictions for that position.
- The predicted probability for each reported metal.

Example return shape:

```python
[
    (12, [('Zn', 0.82), ('Fe', 0.14), ('Mg', 0.03)]),
    (28, [('Fe', 0.76), ('Zn', 0.21), ('Mg', 0.08)])
]
```

## Reproducibility Notes

The dataset split and cross-validation use `random_state=42`. TensorFlow model training can still vary across environments unless TensorFlow, CUDA, cuDNN, hardware, and random seeds are fully controlled.


## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
