
# MAGE: MAsked Generative Encoder to Unify Representation Learning and Image Synthesis

This is an implementation of MAGE: MAsked Generative Encoder to Unify Representation Learning and Image Synthesis 

​The paper titled "MAGE: MAsked Generative Encoder to Unify Representation Learning and Image Synthesis" introduces a novel framework that bridges the gap between two fundamental tasks in computer vision: image generation and representation learning. Traditionally, these tasks have been approached separately, leading to inefficiencies and missed opportunities for mutual enhancement.

![Architecture](./media/architecture.png)

## Idea
MAGE (Masked Generative Encoder) proposes a unified approach that leverages masked image modeling with variable masking ratios to simultaneously train for both generative and representation learning objectives.​

Key Components:

Semantic Tokenization: Utilizes a pre-trained Vector-Quantized Generative Adversarial Network (VQGAN) to convert images into semantic tokens, enabling the model to operate in a discrete latent space.​

Variable Masking Strategy: Applies different masking ratios during training—higher ratios for generative tasks and lower ratios for representation learning—allowing the model to adapt to both objectives within the same framework.​


Encoder-Decoder Architecture: Employs a Vision Transformer (ViT) based encoder-decoder structure to process the masked tokens and reconstruct the original image or learn meaningful representations.​

Contrastive Learning: Incorporates a contrastive loss function at the encoder output to enhance the quality of learned representations, improving performance on downstream tasks.​

Performance Highlights: On the ImageNet-1K dataset, MAGE's ViT-L model achieved a Fréchet Inception Distance (FID) of 9.10 for class-unconditional image generation and a top-1 accuracy of 78.9% for linear probing in representation learning tasks, indicating state-of-the-art performance in both domains. ​


Conclusion: MAGE demonstrates that a single, unified model can effectively handle both image synthesis and representation learning, offering a more efficient and synergistic approach to training in computer vision.

## Available Models

The following models are available with different configurations:

**Large (L) Models:**
- MAGE-ViT-L/4: depth=24, embed_dim=1024, patch_size=4, num_heads=16, decoder_embed_dim=1024 
- MAGE-ViT-L/8: depth=24, embed_dim=1024, patch_size=8, num_heads=16, decoder_embed_dim=1024 
- MAGE-ViT-L/16: depth=24, embed_dim=1024, patch_size=16, num_heads=16, decoder_embed_dim=1024

**Base (B) Models:**
- MAGE-ViT-B/4: depth=12, embed_dim=768, patch_size=4, num_heads=12, decoder_embed_dim=768
- MAGE-ViT-B/8: depth=12, embed_dim=768, patch_size=8, num_heads=12, decoder_embed_dim=768
- MAGE-ViT-B/16: depth=12, embed_dim=768, patch_size=16, num_heads=12, decoder_embed_dim=768
## Model Analysis & Results

### Unconditional Generation
![Unconditional Generation](./media/unconditional_generation.png)

### Linear Probing
![Linear Probing](./media/linear_probing.png)

### Transfer Learning
![Transfer Learning](./media/transfer_learning.png)


## Citation
> **MAGE: MAsked Generative Encoder to Unify Representation Learning and Image Synthesis**  
> *Tianhong Li, Huiwen Chang, Shlok Kumar Mishra, Han Zhang, Dina Katabi, Dilip Krishnan*  
> arXiv 2023 
> [[Paper]](https://arxiv.org/abs/2211.09117)


