# Scandinavian Sentence Transformers

[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)][black]

[black]: https://github.com/psf/black


<!-- start short-description -->

This project is intended for training Danish, Swedish and Norwegian sentence transformers. The project is an extension of the Danish Foundation models project.

<!-- end short-description -->

## Installation

You can install `scandinavian-sentence-transformers` via [pip]:

```bash
git clone {repo url}
cd scandinavian-sentence-transformers
pip install -e .
```

but we recommend using invoke for the setup:
```python
git clone {repo url}
cd scandinavian-sentence-transformers

# install invoke
pip install invoke
# setup up virtual environment and install dependencies
inv setup
```

[pip]: https://pip.pypa.io/en/stable/installing/

## Usage

To train the models you wi
```python
inv prepare_dataset --lang da
inv train --model_name vesteinn/DanskBERT
```
