# First Stage Models Overview

This document provides a comprehensive list of quantizers and VAE models available in the first stage of the autoMED framework. These components are designed to perform specific tasks and can be utilized as building blocks for more complex systems.

## Quantizers in autoMED

The autoMED framework includes various quantizers organized into three main categories: Discrete, Continuous, and Lookup-Free Quantization (LFQ).

### Available Quantizers


- **[GumbelQuantizer](discrete/gumbel_quantizer.py):** Implements Gumbel-Softmax based quantization for discrete latent variables
- **[LookUpFreeQuantizer](https://arxiv.org/html/2310.05737v3):** Language Model Beats Diffusion — Tokenizer is Key to Visual Generation
- **[Finite Scalar Quantization](https://arxiv.org/abs/2309.15505):** Finite Scalar Quantization: VQ-VAE Made Simple
- **[VectorQuantizer2](discrete/vector_quantizer2.py):** Optimized standard VQ-VAE/VQ-GAN quantizer
- **[EMAVectorQuantizer](discrete/ema_vector_quantizer.py):** Standard VQ-VAE/VQ-GAN quantizer with EMA updates
- **[EMAVectorQuantizerWithVAR](discrete/ema_vector_quantizer_with_var.py):** Enhanced version with VAR scaling features

### Available VAEs

- **[VQVAE](http://arxiv.org/abs/1711.00937):** Neural Discrete Representation Learning

- **[VQGAN](https://arxiv.org/abs/2012.09841):** Taming Transformers for High-Resolution Image Synthesis

- **[VAEKL](https://arxiv.org/abs/1312.6114):** Auto-Encoding Variational Bayes

