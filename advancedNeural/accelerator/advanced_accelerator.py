import torch
import torch.nn as nn
import torch.nn.functional as F

class AdvancedAccelerator(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=256, dropout_rate=0.1):
        super(AdvancedAccelerator, self).__init__()

        # Linear layers for initial feature transformation
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        # Multi-Head Self Attention Mechanism
        self.multihead_attn = nn.MultiheadAttention(output_size, num_heads=8, dropout=dropout_rate)

        # Feedforward layer with dropout
        self.feedforward = nn.Sequential(
            nn.Linear(output_size, hidden_size),
            nn.Dropout(p=dropout_rate),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(output_size)
        self.layer_norm2 = nn.LayerNorm(output_size)

        # Dropout for regularization
        self.dropout = nn.Dropout(p=dropout_rate)

        # Residual connection scaling factor
        self.residual_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # Initial feature transformation
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # Multi-Head Self Attention
        attn_output, _ = self.multihead_attn(x, x, x)

        # Residual connection and layer normalization
        x = self.layer_norm1(x + self.residual_scale * self.dropout(attn_output))

        # Feedforward layer
        ff_output = self.feedforward(x)

        # Residual connection and layer normalization
        x = self.layer_norm2(x + self.residual_scale * self.dropout(ff_output))

        return x