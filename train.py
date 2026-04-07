import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader

from config import device, context_length, num_layers, model_dim, num_heads, dropout, batch_size, num_epochs, learning_rate, max_grad_norm
from model import ImageCaptioner
from dataset import download_and_parse_captions, build_vocabulary, ImageCaptioningDataset
from evaluate import compute_bleu, show_example_captions


def validate(model, val_dataloader, loss_function):
    """Run one pass over the validation set and return the average loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, captions in val_dataloader:
            images, captions = images.to(device), captions.to(device)

            captions_in = captions[:, :-1]
            captions_out = captions[:, 1:]
            B, T = captions_in.shape

            preds = model(images, captions_in)
            loss = loss_function(preds.reshape(B * T, -1), captions_out.reshape(B * T))

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def main():
    # --- Data ---
    data_dir, get_captions, all_captions = download_and_parse_captions()
    vocabulary, word_tokenizer = build_vocabulary(all_captions)
    vocab_size = len(vocabulary)

    df = pd.DataFrame(columns=["filename", "caption"])
    df["filename"] = list(get_captions.keys())
    df["caption"] = df["filename"].map(lambda x: get_captions[x])

    # 80 / 10 / 10 split
    n = len(df)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)

    print(f"Split: {len(train_df)} train / {len(val_df)} val / {len(test_df)} test images")

    dataset_kwargs = dict(
        data_dir=data_dir,
        vocabulary=vocabulary,
        word_tokenizer=word_tokenizer,
        context_length=context_length,
    )

    train_dataset = ImageCaptioningDataset(train_df, split="training", **dataset_kwargs)
    val_dataset = ImageCaptioningDataset(val_df, split="validation", **dataset_kwargs)
    test_dataset = ImageCaptioningDataset(test_df, split="validation", **dataset_kwargs)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # --- Model ---
    model = ImageCaptioner(context_length, vocab_size, num_layers, model_dim, num_heads, dropout).to(device)

    # Freeze the CNN encoder
    for param in model.cnn_encoder.parameters():
        param.requires_grad = False

    loss_function = nn.CrossEntropyLoss(ignore_index=vocabulary["<PAD>"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # --- Training ---
    print(f"\nTraining on {device} for {num_epochs} epoch(s).\n")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0

        for images, captions in train_loader:
            images, captions = images.to(device), captions.to(device)

            optimizer.zero_grad()

            captions_in = captions[:, :-1]
            captions_out = captions[:, 1:]
            B, T = captions_in.shape

            preds = model(images, captions_in)
            loss = loss_function(preds.reshape(B * T, -1), captions_out.reshape(B * T))

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

            if num_batches % 100 == 0:
                print(f"  [Epoch {epoch+1}/{num_epochs}] Batch {num_batches} — Train Loss: {loss.item():.4f}")

        avg_train_loss = running_loss / max(num_batches, 1)

        # --- Validation at end of each epoch ---
        avg_val_loss = validate(model, val_loader, loss_function)

        print(f"Epoch {epoch+1}/{num_epochs} — Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # --- Save ---
    print("\nTraining finished!")
    torch.save(model.state_dict(), "captioner_weights.pth")
    print("Weights saved to captioner_weights.pth")

    # --- Test evaluation ---
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)

    avg_test_loss = validate(model, test_loader, loss_function)
    print(f"\nTest Loss: {avg_test_loss:.4f}")

    bleu_score = compute_bleu(model, test_df, data_dir, vocabulary, word_tokenizer, context_length, device)
    print(f"Test BLEU-4: {bleu_score:.4f}")

    print("\n--- Example Captions ---")
    show_example_captions(model, test_df, data_dir, vocabulary, word_tokenizer, context_length, device, num_examples=5)


if __name__ == "__main__":
    main()
