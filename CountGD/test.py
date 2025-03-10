
import torch
import os
def print_model_weights(model, num_samples=5):
    print("Model weights summary:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}:")
            print(f"  Shape: {param.shape}")
            print(f"  dtype: {param.dtype}")
            print(f"  requires_grad: {param.requires_grad}")
            print(f"  Mean: {param.data.mean().item():.6f}")
            print(f"  Std: {param.data.std().item():.6f}")
            print(f"  Min: {param.data.min().item():.6f}")
            print(f"  Max: {param.data.max().item():.6f}")
            if num_samples > 0:
                print(f"  Sample values: {param.data.view(-1)[:num_samples].tolist()}")
            print()

def print_checkpoint_contents(checkpoint_path, print_weights=False, num_samples=5):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print("Contents of checkpoint.pth:")
    for key, value in checkpoint.items():
        if key == "model":
            print("model: Dict containing model state")
            if print_weights:
                print_model_weights(value, num_samples)
        elif isinstance(value, dict):
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, torch.Tensor):
                    print(f"  {sub_key}: Tensor of shape {sub_value.shape}")
                elif isinstance(sub_value, dict):
                    print(f"  {sub_key}: Dict with keys {list(sub_value.keys())}")
                else:
                    print(f"  {sub_key}: {sub_value}")
        elif isinstance(value, torch.Tensor):
            print(f"{key}: Tensor of shape {value.shape}")
        else:
            print(f"{key}: {value}")

# Use the function like this:
checkpoint_path = "C:/Users/syash\Desktop/CountGD/countgd_val/checkpoint.pth"
print_checkpoint_contents(checkpoint_path, print_weights=True, num_samples=5)