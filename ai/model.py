"""
Neural Network Model for Crypto Prediction
Transformer-based architecture for time series forecasting
Professional ML engineering - stability over complexity
"""

import torch
import torch.nn as nn
import math
from typing import Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer
    Adds temporal information to sequence
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            x with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerPredictor(nn.Module):
    """
    Transformer-based model for crypto price prediction

    Architecture:
    1. Input projection layer
    2. Positional encoding
    3. Transformer encoder layers
    4. Global average pooling
    5. Classification head

    Output: 3-class probability distribution [NO TRADE, LONG, SHORT]
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = config.HIDDEN_DIM,
        num_layers: int = config.NUM_LAYERS,
        num_heads: int = config.NUM_HEADS,
        num_classes: int = config.NUM_CLASSES,
        dropout: float = config.DROPOUT,
        sequence_length: int = config.SEQUENCE_LENGTH
    ):
        super().__init__()

        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Input projection: map features to hidden dimension
        self.input_projection = nn.Linear(num_features, hidden_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            d_model=hidden_dim,
            max_len=sequence_length,
            dropout=dropout
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True  # Pre-LN for better stability
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim)
        )

        # Enhanced classification head with deeper architecture
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout / 2),  # Less dropout in final layers
            nn.Linear(hidden_dim // 4, num_classes)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: [batch, seq_len, num_features]

        Returns:
            logits: [batch, num_classes]
        """
        # Input projection
        x = self.input_projection(x)  # [batch, seq_len, hidden_dim]

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoding
        x = self.transformer_encoder(x)  # [batch, seq_len, hidden_dim]

        # Global average pooling over sequence dimension
        x = x.mean(dim=1)  # [batch, hidden_dim]

        # Classification
        logits = self.classifier(x)  # [batch, num_classes]

        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get probability distribution

        Args:
            x: [batch, seq_len, num_features]

        Returns:
            probabilities: [batch, num_classes]
        """
        logits = self.forward(x)
        probabilities = torch.softmax(logits, dim=-1)
        return probabilities


class LSTMPredictor(nn.Module):
    """
    Alternative LSTM-based model
    More stable than Transformer for some datasets
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = config.HIDDEN_DIM,
        num_layers: int = config.NUM_LAYERS,
        num_classes: int = config.NUM_CLASSES,
        dropout: float = config.DROPOUT
    ):
        super().__init__()

        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # Bidirectional for better context
        )

        # Enhanced classification head for LSTM
        # *2 because bidirectional
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize LSTM weights"""
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)

        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: [batch, seq_len, num_features]

        Returns:
            logits: [batch, num_classes]
        """
        # LSTM encoding
        lstm_out, (hidden, cell) = self.lstm(x)  # lstm_out: [batch, seq_len, hidden*2]

        # Take last time step
        last_output = lstm_out[:, -1, :]  # [batch, hidden*2]

        # Classification
        logits = self.classifier(last_output)  # [batch, num_classes]

        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability distribution"""
        logits = self.forward(x)
        probabilities = torch.softmax(logits, dim=-1)
        return probabilities


def create_model(
    num_features: int,
    model_type: str = config.MODEL_TYPE,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> nn.Module:
    """
    Factory function to create model

    Args:
        num_features: Number of input features
        model_type: "transformer" or "lstm"
        device: Device to place model on

    Returns:
        Model instance
    """
    if model_type == "transformer":
        model = TransformerPredictor(num_features=num_features)
    elif model_type == "lstm":
        model = LSTMPredictor(num_features=num_features)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model.to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Created {model_type} model with {num_params:,} parameters")

    return model


if __name__ == "__main__":
    # Test model creation
    num_features = 30
    batch_size = 16
    seq_len = 128

    # Create dummy input
    x = torch.randn(batch_size, seq_len, num_features)

    # Test Transformer
    print("Testing Transformer model:")
    transformer = create_model(num_features, model_type="transformer", device="cpu")
    output = transformer(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    probs = transformer.predict_proba(x)
    print(f"Probabilities shape: {probs.shape}")
    print(f"Sample probabilities: {probs[0]}")
    print(f"Sum: {probs[0].sum()}")

    print("\n" + "="*50 + "\n")

    # Test LSTM
    print("Testing LSTM model:")
    lstm = create_model(num_features, model_type="lstm", device="cpu")
    output = lstm(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    probs = lstm.predict_proba(x)
    print(f"Probabilities shape: {probs.shape}")
    print(f"Sample probabilities: {probs[0]}")
    print(f"Sum: {probs[0].sum()}")
