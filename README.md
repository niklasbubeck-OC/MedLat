# MedTok

A package for autoencoders/tokenizers also applyable for 3D medical data.

## Model registry

All first-stage models (continuous VAEs, discrete VQs, and learned tokenizers) are
registered under a single interface. Instantiate any model by name:

```bash
python examples/load_models.py
```

Or inside your own code:

```python
from src import get_model, available_models

print(available_models("token."))  # list only tokenizers
autoencoder = get_model("continuous.autoencoder.kl-f8")
tokenizer = get_model("token.detok.bb", image_size=256)
```

### Registering new models

Every factory function or class can be exposed automatically by decorating it with
`@register_model("your.prefix.name")`. The decorator works for both functions and
classes and keeps registration consistent across autoencoders, VQs, and tokenizers.
