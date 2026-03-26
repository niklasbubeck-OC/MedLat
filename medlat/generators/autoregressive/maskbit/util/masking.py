"""This file contains the definition of utility functions for masking."""
import math
from typing import Text, Tuple
import torch


def get_mask_tokens(
    tokens: torch.Tensor,
    mask_token: int,
    mode: Text = "arccos",
    min_masking_ratio: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Get the masked tokens.
        Args:
            tokens -> torch.Tensor: The input tokens.
            mask_token -> int: The special `mask` token.
            mode -> Text: The masking function to use (default: "arccos").
        Returns:
            masked_tokens -> torch.Tensor: The masked input tokens. Each masked token is set to mask_token.
            mask -> torch.Tensor: A boolean tensor mask indicating which tokens are masked.
    """
    r = torch.rand(tokens.size(0)) * (1 - min_masking_ratio)
    if mode == "linear":
        val_to_mask = 1 - r
    elif mode == "square":
        val_to_mask = 1 - (r ** 2)
    elif mode == "cosine":
        val_to_mask = torch.cos(r * math.pi * 0.5)
    elif mode == "arccos":
        val_to_mask = torch.acos(r) / (math.pi * 0.5)
    else:
        raise ValueError("Invalid mode. Choose between 'linear','square', 'cosine', 'arccos'.")
    
    masked_tokens = tokens.detach().clone()
    mask = torch.rand(tokens.size()) < val_to_mask.view(-1, 1, 1)

    masked_tokens[mask] = torch.full_like(masked_tokens[mask], mask_token)
    return masked_tokens, mask


def get_masking_ratio(progress: float, mode: Text = "arccos") -> torch.Tensor:
    """ Get masking ratio. 
        Args:
            progress -> float: The percentage of iterations already done.
            mode -> Text: The masking function to use (default: "arccos").

        Returns:
            val_to_mask -> torch.Tensor: The masking ratio.
    """
    r = torch.tensor(progress)
    if mode == "root":
        val_to_mask = 1 - (r ** 0.5)
    elif mode == "square":
        val_to_mask = 1 - (r ** 2)
    elif mode == "cosine":
        val_to_mask = torch.cos(r * math.pi * 0.5)
    elif mode == "arccos":
        val_to_mask = torch.acos(r) / (math.pi * 0.5)
    elif mode == "linear":
        val_to_mask = 1 - r
    else:
        raise ValueError("Invalid mode. Choose between 'linear','square', 'cosine', 'arccos', 'root'.")
    
    val_to_mask = torch.clamp(val_to_mask, 1e-6, 1.0)
    return val_to_mask

def get_exact_mask(shape: torch.Size, masking_ratio: float) -> torch.Tensor:
    """
    Generate a fixed binary mask with an exact number of masked tokens.

    Args:
        shape -> torch.Size: Shape of the mask (same as input tokens).
        masking_ratio -> float: The fixed ratio of tokens to mask.

    Returns:
        mask -> torch.Tensor: A boolean tensor indicating which tokens are masked.
    """
    assert 0.0 <= masking_ratio <= 1.0, "Masking ratio must be between 0 and 1"

    batch_size, num_tokens, splits = shape  # Extract dimensions
    num_masked_tokens = int(num_tokens * masking_ratio * splits)  # Exact number to mask

    # Generate random values for each token
    rand_vals = torch.rand((batch_size, num_tokens, splits))

    # Flatten for sorting
    rand_vals_flat = rand_vals.view(batch_size, -1)

    # Select the top K values to mask
    _, topk_indices = torch.topk(rand_vals_flat, num_masked_tokens, largest=False)  # Smallest values get masked

    # Initialize the mask as all False
    mask = torch.zeros_like(rand_vals_flat, dtype=torch.bool)

    # Set selected indices to True (masked)
    mask.scatter_(1, topk_indices, True)

    # Reshape back to original shape
    mask = mask.view(batch_size, num_tokens, splits)

    return mask


if __name__ == "__main__":
    # Test the masking functions
    tokens = torch.randint(0, 100, (1, 12, 2))
    exact_mask = get_exact_mask(tokens.shape, 0.75)

    num_masked_elements = exact_mask.sum().item()
    total_elements = tokens.numel()
    inferred_ratio = num_masked_elements / total_elements
    print(f"Number of masked elements: {num_masked_elements}")
    print(f"Inferred masking ratio: {inferred_ratio:.4f}")
    print(f"Expected masking ratio: {0.75}")

    masked_tokens = tokens.clone()
    masked_tokens[exact_mask] = -1
    print("Masked tokens:")
    print(masked_tokens)
    print("Mask:")
    print(exact_mask)
    print("Original tokens:")
    print(tokens)
