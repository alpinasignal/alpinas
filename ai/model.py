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


class AttentionPooling(nn.Module):
    """
    Learnable attention-based pooling
    Better than mean pooling for capturing important temporal features
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, hidden_dim]
        Returns:
            pooled: [batch, hidden_dim]
        """
        # Calculate attention weights
        attn_weights = self.attention(x)  # [batch, seq_len, 1]
        attn_weights = torch.softmax(attn_weights, dim=1)  # [batch, seq_len, 1]

        # Weighted sum
        pooled = torch.sum(x * attn_weights, dim=1)  # [batch, hidden_dim]
        return pooled


class TransformerPredictor(nn.Module):
    """
    Transformer-based model for crypto price prediction

    Architecture:
    1. Input projection layer
    2. Positional encoding
    3. Transformer encoder layers
    4. Attention-based pooling (learnable)
    5. Advanced classification head with residual connections

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

        # Attention-based pooling instead of simple mean
        self.attention_pool = AttentionPooling(hidden_dim)

        # Maximum accuracy classification head with residual connections
        self.classifier_layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.classifier_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(dropout)

        self.classifier_layer3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.norm3 = nn.LayerNorm(hidden_dim // 2)
        self.dropout3 = nn.Dropout(dropout)

        self.classifier_layer4 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.norm4 = nn.LayerNorm(hidden_dim // 4)
        self.dropout4 = nn.Dropout(dropout / 2)

        self.classifier_output = nn.Linear(hidden_dim // 4, num_classes)

        # Residual projection for skip connection
        self.residual_proj = nn.Linear(hidden_dim, hidden_dim)

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
        Forward pass with attention pooling and residual connections

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

        # Attention-based pooling (learnable)
        x = self.attention_pool(x)  # [batch, hidden_dim]

        # Store for residual connection
        identity = x

        # Classification with residual connections
        # Layer 1 with residual
        x = self.classifier_layer1(x)
        x = self.norm1(x)
        x = nn.functional.gelu(x)
        x = self.dropout1(x)
        x = x + self.residual_proj(identity)  # Residual connection

        # Layer 2 with residual
        identity2 = x
        x = self.classifier_layer2(x)
        x = self.norm2(x)
        x = nn.functional.gelu(x)
        x = self.dropout2(x)
        x = x + identity2  # Residual connection

        # Layer 3
        x = self.classifier_layer3(x)
        x = self.norm3(x)
        x = nn.functional.gelu(x)
        x = self.dropout3(x)

        # Layer 4
        x = self.classifier_layer4(x)
        x = self.norm4(x)
        x = nn.functional.gelu(x)
        x = self.dropout4(x)

        # Output layer
        logits = self.classifier_output(x)  # [batch, num_classes]

        return logits

    def predict_proba(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Get probability distribution with optional temperature scaling

        Temperature scaling calibrates confidence:
        - temperature > 1: More uniform (less confident)
        - temperature < 1: More peaked (more confident)
        - temperature = 1: No change

        Args:
            x: [batch, seq_len, num_features]
            temperature: Calibration temperature (default 1.0)

        Returns:
            probabilities: [batch, num_classes]
        """
        logits = self.forward(x)
        # Apply temperature scaling for calibration
        scaled_logits = logits / temperature
        probabilities = torch.softmax(scaled_logits, dim=-1)
        return probabilities

    def get_confidence_calibrated(self, x: torch.Tensor, temperature: float = 1.2) -> torch.Tensor:
        """
        Get calibrated probabilities with slight smoothing
        Prevents overconfident predictions

        Args:
            x: Input tensor
            temperature: Calibration temperature (>1 for smoother probs)

        Returns:
            Calibrated probabilities
        """
        return self.predict_proba(x, temperature=temperature)


class LSTMPredictor(nn.Module):
    """
    Alternative LSTM-based model
    Enhanced with attention pooling and residual connections
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

        # Attention pooling for LSTM output
        self.attention_pool = AttentionPooling(hidden_dim * 2)  # *2 for bidirectional

        # Maximum accuracy classification head with residual connections
        # *2 because bidirectional
        self.classifier_layer1 = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.norm1 = nn.LayerNorm(hidden_dim * 2)
        self.dropout1 = nn.Dropout(dropout)

        self.classifier_layer2 = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.norm2 = nn.LayerNorm(hidden_dim * 2)
        self.dropout2 = nn.Dropout(dropout)

        self.classifier_layer3 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout3 = nn.Dropout(dropout)

        self.classifier_layer4 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.norm4 = nn.LayerNorm(hidden_dim // 2)
        self.dropout4 = nn.Dropout(dropout / 2)

        self.classifier_output = nn.Linear(hidden_dim // 2, num_classes)

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
        Forward pass with attention pooling and residual connections

        Args:
            x: [batch, seq_len, num_features]

        Returns:
            logits: [batch, num_classes]
        """
        # LSTM encoding
        lstm_out, (hidden, cell) = self.lstm(x)  # lstm_out: [batch, seq_len, hidden*2]

        # Attention-based pooling instead of just last timestep
        x = self.attention_pool(lstm_out)  # [batch, hidden*2]

        # Store for residual connection
        identity = x

        # Classification with residual connections
        # Layer 1 with residual
        x = self.classifier_layer1(x)
        x = self.norm1(x)
        x = nn.functional.relu(x)
        x = self.dropout1(x)
        x = x + identity  # Residual connection

        # Layer 2 with residual
        identity2 = x
        x = self.classifier_layer2(x)
        x = self.norm2(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = x + identity2  # Residual connection

        # Layer 3
        x = self.classifier_layer3(x)
        x = self.norm3(x)
        x = nn.functional.relu(x)
        x = self.dropout3(x)

        # Layer 4
        x = self.classifier_layer4(x)
        x = self.norm4(x)
        x = nn.functional.relu(x)
        x = self.dropout4(x)

        # Output layer
        logits = self.classifier_output(x)  # [batch, num_classes]

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
