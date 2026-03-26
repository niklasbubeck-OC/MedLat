"""This file contains the definition of the sampling function for pseudo-3D models."""

from typing import Optional, Tuple, List, Text

import torch

from ..pseudo_3d import Pseudo3DWrapper
from .masking import get_masking_ratio
from .factorization import combine_factorized_tokens


@torch.no_grad()
def sample(
    model: Pseudo3DWrapper,
    num_samples: int = 2,
    labels: Optional[torch.Tensor] = None,
    softmax_temperature: float = 1.0,
    randomize_temperature: float = 4.5,
    mask_schedule_strategy: Text = "linear",
    num_steps: int = 12,
    guidance_scale: float = 3.0,
    guidance_annealing: Text = "none",
    use_sampling_annealing: bool = False,
    scale_pow: float = 4.0,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Sample from the pseudo-3D model.

    Args:
        model -> Pseudo3DWrapper: The wrapper model containing both MLM and VQ models.
        num_samples -> int: The number of samples to generate.
        labels -> Optional[torch.Tensor]: The labels to use for the generation.
        softmax_temperature -> float: The temperature for the softmax.
        randomize_temperature -> float: The temperature for the randomization.
        mask_schedule_strategy -> Text: The strategy for the mask schedule.
        num_steps -> int: The number of steps to use for the sampling.
        guidance_scale -> float: The scale for the guidance.
        guidance_annealing -> Text: The annealing strategy for the guidance.
        use_sampling_annealing -> bool: Whether to use the sampling annealing.
        scale_pow -> float: The power for the scaling.

    Returns:
        Tuple[torch.Tensor, List[torch.Tensor]]: The generated samples and the tokens at each step.
    """
    device = model.device
    mlm_model = model.mlm
    mask_token = mlm_model.mask_token
    patch_size = model.patch_size
    codebook_size = mlm_model.effective_codebook_size
    codebook_splits = mlm_model.splits
    num_time_steps = mlm_model.num_time_steps

    mlm_model.eval()

    if labels is None:
        labels = [0] * num_samples
        labels = torch.LongTensor(labels).to(device)

    drop_labels = torch.ones(num_samples, dtype=bool, device=device)
    spatial_size = int(patch_size ** 2)
    num_splits = int(codebook_splits)
    
    # Initialize tokens with time dimension
    masked_tokens = torch.full((num_samples, num_time_steps * spatial_size, num_splits), mask_token, device=device)
    num_maskable = num_time_steps * spatial_size * num_splits
    mask = (masked_tokens == mask_token)
    num_sampled = torch.zeros_like(masked_tokens, dtype=torch.int)
    l_full_tokens = []
    gumbel = torch.distributions.Gumbel(loc=0.0, scale=1.0)

    for i in range(num_steps):
        progress = (i + 1) / num_steps
        if guidance_scale != 0.0:
            logits = mlm_model(
                torch.cat([masked_tokens.clone(), masked_tokens.clone()], dim=0),
                torch.cat([labels, labels], dim=0),
                torch.cat([~drop_labels, drop_labels], dim=0)
            )
            # Classifier-free guidance
            logits_with_class, logits_without_class = torch.chunk(logits, 2, dim=0)
            if guidance_annealing == "none":
                scale_step = 1.0
            elif guidance_annealing == "linear":
                scale_step = i / num_steps
            elif guidance_annealing == "cosine":
                scale_pow = torch.ones((1), device=device) * scale_pow
                scale_step = (1 - torch.cos(((i / num_steps) ** scale_pow) * torch.pi)) * 1/2 # power-cos scaling
            scale = guidance_scale * scale_step
            logits = logits_with_class + scale * (logits_with_class - logits_without_class)
        else:
            # ! We are not inversing the drop_labels mask here, because we don't use the classes for the guidance
            logits = mlm_model(masked_tokens.clone(), labels, drop_labels) 
        
        if use_sampling_annealing:
            softmax_temperature = 0.5 + 0.8 * (1 - progress)
        probabilities = torch.softmax(logits / softmax_temperature, dim=-1)
        distribution = torch.distributions.Categorical(probabilities)
        predicted_tokens = distribution.sample()

        num_masked = torch.sum(mask, dim=(1,2))[0]

        predicted_tokens = torch.where(mask, predicted_tokens, masked_tokens)

        confidence = torch.gather(probabilities, -1, predicted_tokens.unsqueeze(-1)).squeeze(-1)
        # Ignore existing tokens by overwriting the confidence.
        confidence = torch.where(mask, confidence, torch.inf)

        noise = gumbel.sample(predicted_tokens.size()) * randomize_temperature * (1 - progress)
        confidence = torch.log(confidence) + noise.to(device)

        mask_ratio = get_masking_ratio(progress, mode=mask_schedule_strategy).to(device)
        
        # min = 1, max = num_masked - 1
        mask_len = torch.floor(mask_ratio * num_maskable)
        num_tokens_to_mask = torch.clamp(mask_len, torch.ones_like(num_masked), num_masked-1).long()
        sorted_confidence = torch.sort(confidence.view(num_samples, -1), dim=-1).values
        threshold = sorted_confidence[:, num_tokens_to_mask - 1]

        should_mask = (confidence <= threshold.unsqueeze(-1).unsqueeze(-1))
        masked_tokens = torch.where(should_mask, mask_token, predicted_tokens)
        mask = (masked_tokens == mask_token)
        num_sampled += torch.where(should_mask, 0, 1)
        l_full_tokens.append(predicted_tokens)

    predicted_tokens = combine_factorized_tokens(predicted_tokens, codebook_size, codebook_splits)
    
    # Use the wrapper's vq_decode method
    generated_image = model.vq_decode(predicted_tokens)
    
    return generated_image, l_full_tokens