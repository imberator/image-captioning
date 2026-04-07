import torch
import torch.nn as nn
import timm


class ImageCaptioner(nn.Module):

  def __init__(self, context_length, vocab_size, num_blocks, model_dim, num_heads, dropout):
    super().__init__()
    self.cnn_encoder = timm.create_model("efficientnet_b0", pretrained=True)
    
    # Determine in_features dynamically
    test_image = torch.randn(1, 3, 224, 224)
    self.cnn_encoder.eval()
    with torch.no_grad():
      cnn_features = self.cnn_encoder(test_image)
    in_features = cnn_features.shape[1]

    self.project = nn.Linear(in_features, model_dim)

    self.word_embeddings = nn.Embedding(vocab_size, model_dim)
    self.pos_embeddings = nn.Embedding(context_length, model_dim)

    layer = nn.TransformerDecoderLayer(
        model_dim, num_heads, 2 * model_dim, dropout=dropout,
        batch_first=True, norm_first=True)
    self.layers = nn.TransformerDecoder(layer, num_blocks)

    self.vocab_projection = nn.Linear(model_dim, vocab_size)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, labels):
    tok_embedded = self.word_embeddings(labels)
    B, T = labels.shape
    pos_embedded = self.pos_embeddings(torch.arange(T, device=labels.device))
    embedded = tok_embedded + pos_embedded
    
    embedded = self.dropout(embedded)

    self.cnn_encoder.eval()
    with torch.no_grad():
      cnn_out = self.cnn_encoder(x)
      
    # Project outside of torch.no_grad() so the linear layer can get gradients and train.
    encoded_img = self.project(cnn_out.view(B, -1))

    # Captions -> [B, T, D]
    # Convert images from [B, D] -> [B, 1, D]
    encoded_img = encoded_img.unsqueeze(1)

    attn_mask = nn.Transformer.generate_square_subsequent_mask(T).to(labels.device)
    
    tgt_pad_mask = (labels == 1).float()
    
    layers_out = self.layers(tgt=embedded, memory=encoded_img, tgt_mask=attn_mask, tgt_key_padding_mask=tgt_pad_mask)
    
    logits = self.vocab_projection(layers_out)  # (B, T, vocab_size)
    
    return logits

  @torch.no_grad()
  def generate_caption(self, image, start_token_idx, end_token_idx, max_length):
    """Autoregressively generate a caption for a single image.
    
    Args:
        image: Preprocessed image tensor of shape (1, 3, 224, 224).
        start_token_idx: Integer index of the <START> token in the vocabulary.
        end_token_idx: Integer index of the <END> token in the vocabulary.
        max_length: Maximum number of tokens to generate.

    Returns:
        List of token indices (excluding <START>).
    """
    self.eval()
    device = image.device

    # Encode the image once
    self.cnn_encoder.eval()
    cnn_out = self.cnn_encoder(image)
    encoded_img = self.project(cnn_out.view(1, -1)).unsqueeze(1)  # (1, 1, D)

    # Start with just the <START> token
    generated = [start_token_idx]

    for _ in range(max_length):
      tokens = torch.tensor([generated], dtype=torch.long, device=device)  # (1, seq_len)
      T = tokens.shape[1]

      tok_emb = self.word_embeddings(tokens)
      pos_emb = self.pos_embeddings(torch.arange(T, device=device))
      embedded = tok_emb + pos_emb

      attn_mask = nn.Transformer.generate_square_subsequent_mask(T).to(device)
      out = self.layers(tgt=embedded, memory=encoded_img, tgt_mask=attn_mask)
      logits = self.vocab_projection(out[:, -1, :])  # (1, vocab_size)

      next_token = logits.argmax(dim=-1).item()
      generated.append(next_token)

      if next_token == end_token_idx:
        break

    # Return everything after <START>, excluding <END> if present
    result = generated[1:]
    if result and result[-1] == end_token_idx:
      result = result[:-1]
    return result