from math import isqrt

from torch import nn


class CNNModel(nn.Module):
    def __init__(self, square_size=40):
        super().__init__()
        if square_size not in {32, 40}:
            raise ValueError("Supported checkpoint input sizes are 32 or 40 pixels per square.")
        self.square_size = square_size
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Keep the original layer names for checkpoint compatibility.
        self.out = nn.Linear(32 * (square_size // 4) ** 2, 13)

    @classmethod
    def from_state_dict(cls, state):
        outputs, features = state["out.weight"].shape
        side = isqrt(features // 32)
        if outputs != 13 or features != 32 * side**2:
            raise ValueError("The checkpoint does not match the 13-class CNN architecture.")
        model = cls(square_size=side * 4)
        model.load_state_dict(state)
        return model

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.flatten(start_dim=1)
        logits = self.out(x)
        return logits
