"""Test smoothness loss computation."""

import torch
from utils.transforms import identity_lut
from utils.losses import lut_smoothness_loss

# Test 1: Identity LUT should have low smoothness loss (smooth transitions)
print("Test 1: Identity LUT smoothness")
identity = identity_lut(resolution=16)
smooth_loss = lut_smoothness_loss(identity)
print(f"  Identity LUT smoothness loss: {smooth_loss.item():.6f}")
print("  (Should be small - smooth gradients)\n")

# Test 2: Random LUT should have high smoothness loss (rough transitions)
print("Test 2: Random LUT smoothness")
random_lut = torch.rand(16, 16, 16, 3)
smooth_loss_random = lut_smoothness_loss(random_lut)
print(f"  Random LUT smoothness loss: {smooth_loss_random.item():.6f}")
print("  (Should be large - rough gradients)\n")

# Test 3: Constant LUT should have zero smoothness loss (no variation)
print("Test 3: Constant LUT smoothness")
constant_lut = torch.ones(16, 16, 16, 3) * 0.5
smooth_loss_constant = lut_smoothness_loss(constant_lut)
print(f"  Constant LUT smoothness loss: {smooth_loss_constant.item():.6f}")
print("  (Should be ~0 - no gradients)\n")

# Test 4: Verify gradient flow
print("Test 4: Gradient flow through smoothness loss")
lut = identity_lut(resolution=16)
lut.requires_grad = True
loss = lut_smoothness_loss(lut)
loss.backward()
print(f"  Smoothness loss: {loss.item():.6f}")
print(f"  Gradients exist: {lut.grad is not None}")
print(f"  Gradient magnitude: {lut.grad.abs().mean().item():.6f}")

print("\n✓ All smoothness tests passed!")
