# Vanilla Transformer for English to German Translation

This repository contains an implementation of a vanilla Transformer model for the task of English-to-German translation. The Transformer architecture is implemented from scratch and trained on the IWSLT 2017 TED Talk dataset.

## Dataset

* **Source**: IWSLT 2017 TED Talk dataset
* **Training Files**: `english_train.txt` (English), `german_train.txt` (German)
* **Validation Files**:

  * Full: `english_val_all`, `german_val_all`
  * Subset: `english_val`, `german_val`

> Note: The dataset has been preprocessed by converting text to token sequences to reduce training time. All file paths referring to English use the term `encodings`, and German uses `decodings`.

## Tokenizer

* **Type**: Byte Pair Encoding (BPE)
* **Inspiration**: Andrej Karpathy's YouTube explanation ([Watch here](https://www.youtube.com/watch?v=zduSFxRajkE))
* **Tokenizer Script**: `tokenizer.py`
* **Vocabulary and Merge Dictionary**: Stored in `vocab.pkl`
* **Related Files**: Located in the `data_files/` folder

## Model

* **Implementation**: `model.py`
* **Architecture**: Standard vanilla Transformer with encoder-decoder structure

## Training

* **Script**: `train.py`
* **Features**:

  * Train the model from scratch
  * Resume training from a checkpoint

## Validation and Inference

* **Validation Script**: `validate.py`

  * Visualizes the model's translation performance on the validation set

* **Inference Script**: `infer.py`

  * Translates unseen English sentences to German using the trained model

## Folder Structure

```
.
├── data_files/           # Contains preprocessed datasets and vocab.pkl
├── tokenizer.py          # BPE tokenizer implementation
├── model.py              # Transformer model architecture
├── my_utils.py           # Utility functions like Multi Head Attention, Positional Encodings
├── train.py              # Script for training the model
├── validate.py           # Script for validating and visualizing performance
├── infer.py              # Script for running inference on unseen text
```

## Usage

Train the model:

```bash
python train.py
```

Validate model performance:

```bash
python validate.py
```

Run inference:

```bash
python infer.py --input_sentence "Your English sentence here"
```

---

For any questions or contributions, please open an issue or submit a pull request.
