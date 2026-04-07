import os
import torch
import cv2
import albumentations as alb
from albumentations.pytorch import ToTensorV2
from collections import Counter


def _preprocess_image(img_path):
    """Load and preprocess a single image for inference."""
    raw_img = cv2.imread(img_path)
    if raw_img is None:
        raise FileNotFoundError(f"Could not find image at {img_path}")
    
    actual_image = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

    transform = alb.Compose([
        alb.Resize(224, 224),
        alb.Normalize(),
        ToTensorV2(),
    ])
    
    return transform(image=actual_image)["image"].unsqueeze(0)  # (1, 3, 224, 224)


def _decode_tokens(token_indices, vocabulary):
    """Convert a list of token indices back to a list of words."""
    # Build reverse lookup: index -> word
    itos = vocabulary.get_itos()
    words = []
    for idx in token_indices:
        word = itos[idx]
        if word in ("<UNKNOWN>", "<PAD>", "<START>", "<END>"):
            continue
        words.append(word)
    return words


def compute_bleu(model, test_df, data_dir, vocabulary, word_tokenizer, context_length, device):
    """Compute corpus-level BLEU-4 score over the test set.

    For each test image, generates one caption using greedy decoding and
    compares it against all reference captions for that image.
    
    Returns:
        BLEU-4 score as a float between 0 and 1.
    """
    model.eval()

    start_idx = vocabulary["<START>"]
    end_idx = vocabulary["<END>"]

    all_references = []   # List of List[List[str]]  (one entry per image)
    all_hypotheses = []   # List of List[str]

    for i in range(len(test_df)):
        image_filename, reference_captions = test_df.iloc[i]
        img_path = os.path.join(data_dir, "Images", image_filename)

        try:
            image = _preprocess_image(img_path).to(device)
        except FileNotFoundError:
            continue

        # Generate caption
        generated_indices = model.generate_caption(image, start_idx, end_idx, max_length=context_length)
        hypothesis = _decode_tokens(generated_indices, vocabulary)

        # Tokenize all reference captions for this image
        refs = [word_tokenizer(cap) for cap in reference_captions]

        all_hypotheses.append(hypothesis)
        all_references.append(refs)

    # Compute BLEU-4 using a simple implementation
    bleu = _corpus_bleu(all_references, all_hypotheses, max_n=4)
    return bleu


def _corpus_bleu(all_references, all_hypotheses, max_n=4):
    """Compute corpus-level BLEU score (without external dependencies).

    Args:
        all_references: List of reference lists. Each entry is a list of
            reference token lists for one example.
        all_hypotheses: List of hypothesis token lists.
        max_n: Maximum n-gram order (default 4 for BLEU-4).

    Returns:
        BLEU score as a float.
    """
    import math

    clipped_counts = Counter()
    total_counts = Counter()
    hyp_lengths = 0
    ref_lengths = 0

    for refs, hyp in zip(all_references, all_hypotheses):
        hyp_lengths += len(hyp)
        # Pick the reference closest in length to the hypothesis
        ref_lengths += min(len(r) for r in refs) if refs else 0

        for n in range(1, max_n + 1):
            # Count n-grams in hypothesis
            hyp_ngrams = Counter()
            for i in range(len(hyp) - n + 1):
                ngram = tuple(hyp[i:i + n])
                hyp_ngrams[ngram] += 1

            # Max count of each n-gram across all references
            max_ref_ngrams = Counter()
            for ref in refs:
                ref_ngrams = Counter()
                for i in range(len(ref) - n + 1):
                    ngram = tuple(ref[i:i + n])
                    ref_ngrams[ngram] += 1
                for ngram, count in ref_ngrams.items():
                    max_ref_ngrams[ngram] = max(max_ref_ngrams[ngram], count)

            # Clipped counts
            for ngram, count in hyp_ngrams.items():
                clipped_counts[n] += min(count, max_ref_ngrams.get(ngram, 0))
                total_counts[n] += count

    # Compute precision for each n-gram order
    precisions = []
    for n in range(1, max_n + 1):
        if total_counts[n] == 0:
            return 0.0
        precisions.append(clipped_counts[n] / total_counts[n])

    # Geometric mean of precisions
    log_avg = sum(math.log(p) for p in precisions if p > 0) / max_n

    # Brevity penalty
    if hyp_lengths == 0:
        return 0.0
    bp = math.exp(min(0, 1 - ref_lengths / hyp_lengths))

    return bp * math.exp(log_avg)


def show_example_captions(model, test_df, data_dir, vocabulary, word_tokenizer, context_length, device, num_examples=5):
    """Print generated vs reference captions for a few test images."""
    model.eval()

    start_idx = vocabulary["<START>"]
    end_idx = vocabulary["<END>"]

    shown = 0
    for i in range(len(test_df)):
        if shown >= num_examples:
            break

        image_filename, reference_captions = test_df.iloc[i]
        img_path = os.path.join(data_dir, "Images", image_filename)

        try:
            image = _preprocess_image(img_path).to(device)
        except FileNotFoundError:
            continue

        generated_indices = model.generate_caption(image, start_idx, end_idx, max_length=context_length)
        generated_words = _decode_tokens(generated_indices, vocabulary)
        generated_sentence = " ".join(generated_words)

        print(f"\n  Image: {image_filename}")
        print(f"  Generated:  {generated_sentence}")
        print(f"  Reference:  {reference_captions[0]}")
        shown += 1


if __name__ == "__main__":
    """Standalone evaluation — load saved weights and evaluate on the test set."""
    import pandas as pd
    from config import device, context_length, num_layers, model_dim, num_heads, dropout
    from model import ImageCaptioner
    from dataset import download_and_parse_captions, build_vocabulary

    data_dir, get_captions, all_captions = download_and_parse_captions()
    vocabulary, word_tokenizer = build_vocabulary(all_captions)
    vocab_size = len(vocabulary)

    df = pd.DataFrame(columns=["filename", "caption"])
    df["filename"] = list(get_captions.keys())
    df["caption"] = df["filename"].map(lambda x: get_captions[x])

    # Reproduce the same 80/10/10 split
    n = len(df)
    test_df = df.iloc[int(0.9 * n):].reset_index(drop=True)

    model = ImageCaptioner(context_length, vocab_size, num_layers, model_dim, num_heads, dropout).to(device)
    model.load_state_dict(torch.load("captioner_weights.pth", map_location=device))
    print("Loaded weights from captioner_weights.pth")

    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)

    bleu_score = compute_bleu(model, test_df, data_dir, vocabulary, word_tokenizer, context_length, device)
    print(f"\nBLEU-4: {bleu_score:.4f}")

    print("\n--- Example Captions ---")
    show_example_captions(model, test_df, data_dir, vocabulary, word_tokenizer, context_length, device, num_examples=10)
