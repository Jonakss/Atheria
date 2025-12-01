import torch
import torch.nn as nn

class MockModel(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Input: [batch, channels, H, W]
        # Output: [batch, channels, H, W]
        # Simple identity-like convolution
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        
        # Initialize to identity-ish to keep particles alive
        nn.init.dirac_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        
    def forward(self, x):
        return self.conv(x)

# d_state=64 means 128 channels (real + imag)
model = MockModel(channels=128)
model.eval()

# Trace it
example_input = torch.randn(1, 128, 64, 64)
traced_model = torch.jit.trace(model, example_input)

traced_model.save("dummy_native_model_128ch.pt")
print("Saved dummy_native_model_128ch.pt")
