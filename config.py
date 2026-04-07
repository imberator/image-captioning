import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model architecture
context_length = 25
num_layers = 4
model_dim = 256
num_heads = 8
dropout = 0.1

# Training
batch_size = 64
num_epochs = 20
learning_rate = 3e-4
max_grad_norm = 1.0
