# Fractal Generative Models

This is an implementation of [Fractal Generative Models](https://arxiv.org/pdf/2502.17437).

![main figure](./media/main_figure.PNG)


## TODO: 
- [ ] 



## Main Idea
The paper introduces a novel generative modeling framework inspired by fractals, called Fractal Generative Models. The key concept is to recursively construct generative models using smaller atomic generative modules, resulting in self-similar, hierarchical architectures. This recursive approach mirrors the structure of fractals in mathematics, where complex patterns emerge from simple, repeated rules.

The framework leverages the modularization of generative models (e.g., autoregressive models) to create a scalable and efficient system for modeling high-dimensional data distributions, such as images.

## Available Models
The following are available models as presented in the paper
- FractalAR_in64
- FractalMAR_in64
- FractalMAR_base_in256
- FractalMAR_large_in256
- FractalMAR_huge_in256


## Model Analysis & Results

### Computational Costs
![Computational Costs](./media/computational_costs.PNG)


### Likelihood Estimation
![Likelihood Estimation](./media/likelihood_estimation.PNG)

### Sample Quality
![Sample Quality](./media/sample_quality.PNG)


## Citation
> **Fractal Generative Models**  
> *Tianhong Li, Qinyi Sun, Lijie Fan, Kaiming He*  
> arXiv 2025  
> [[Paper]](https://arxiv.org/pdf/2502.17437)
