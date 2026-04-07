# Image Captioning

A multimodal image captioning model that generates natural language descriptions for images. Uses a frozen **EfficientNet-B0** encoder and a **Transformer Decoder** trained on the [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) dataset.

## Architecture

| Component | Details |
|---|---|
| Encoder | EfficientNet-B0 (frozen, pre-trained on ImageNet) |
| Projection | Linear layer mapping CNN features → decoder dimension |
| Decoder | 4-layer Transformer Decoder with causal masking |
| Training | Causal language modeling with teacher forcing |

## Project Structure

```
├── config.py        # Hyperparameters and device configuration
├── model.py         # ImageCaptioner architecture + greedy caption generation
├── dataset.py       # Data downloading (via kagglehub), parsing, vocabulary, Dataset class
├── train.py         # Training loop with per-epoch validation and test evaluation
├── evaluate.py      # BLEU-4 scoring and example caption generation
└── requirements.txt
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Train
```bash
python train.py
```

The Flickr8k dataset is automatically downloaded on first run via `kagglehub` and cached locally. Training runs for the number of epochs defined in `config.py`, with validation after each epoch and a full test evaluation (BLEU-4 + example captions) at the end.

### Evaluate
To re-evaluate saved weights without retraining:
```bash
python evaluate.py
```

## Configuration

All hyperparameters are in `config.py`