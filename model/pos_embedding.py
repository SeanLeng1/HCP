import torch
from torch import nn

class PositionalEmbedding(nn.Module):
    def __init__(self, hidden_dim, max_seq_len, device = 'cpu'):
        super().__init__()
        self.encoding = torch.zeros(max_seq_len, hidden_dim, device=device)
        self.encoding.requires_grad = False  # no need to backpropagate

        pos = torch.arange(0, max_seq_len, device=device)
        pos = pos.float().unsqueeze(1)

        _2i = torch.arange(0, hidden_dim, step = 2, device = device).float()

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / hidden_dim)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / hidden_dim)))

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        embed = self.encoding[:seq_len, :]
        return x + embed
    

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_seq_len, device = 'cpu'):
        super(LearnablePositionalEncoding, self).__init__()
        # Define a learnable embedding with max_len and d_model
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim).to(device)

    def forward(self, x):
        """
        Forward pass of the learnable positional encoding.

        :param x: input tensor with shape [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, hidden_dim = x.shape
        pos = torch.arange(0, seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)
        pos_encoded = self.pos_embedding(pos)
        x = x + pos_encoded
        return x
