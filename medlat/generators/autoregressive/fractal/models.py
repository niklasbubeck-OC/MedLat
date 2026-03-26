import torch 
from .fractalgen import FractalGen

__all__ = [
    "FractalAR_in64",
    "FractalMAR_in64",
    "FractalMAR_base_in256",
    "FractalMAR_large_in256",
    "FractalMAR_huge_in256"
]

class FractalAR_in64(FractalGen):
    def __init__(self, *args, **kwargs):
        super().__init__(
            img_size_list=(64, 4, 1),
            embed_dim_list=(1024, 512, 128),
            num_blocks_list=(32, 8, 3),
            num_heads_list=(16, 8, 4),
            generator_type_list=("ar", "ar", "ar"),
            fractal_level=0,
            **kwargs
            )

class FractalMAR_in64(FractalGen):
    def __init__(self, *args, **kwargs):
        super().__init__(
            img_size_list=(64, 4, 1),
            embed_dim_list=(1024, 512, 128),
            num_blocks_list=(32, 8, 3),
            num_heads_list=(16, 8, 4),
            generator_type_list=("mar", "mar", "ar"),
            fractal_level=0,
            **kwargs
            )

class FractalMAR_base_in256(FractalGen):
    def __init__(self, *args, **kwargs):
        super().__init__(
            img_size_list=(256, 16, 4, 1),
            embed_dim_list=(768, 384, 192, 64),
            num_blocks_list=(24, 6, 3, 1),
            num_heads_list=(12, 6, 3, 4),
            generator_type_list=("mar", "mar", "mar", "ar"),
            fractal_level=0,
            **kwargs
            )

class FractalMAR_large_in256(FractalGen):
    def __init__(self, *args, **kwargs):
        super().__init__(
            img_size_list=(256, 16, 4, 1),
            embed_dim_list=(1024, 512, 256, 64),
            num_blocks_list=(32, 8, 4, 1),
            num_heads_list=(16, 8, 4, 4),
            generator_type_list=("mar", "mar", "mar", "ar"),
            fractal_level=0,
            **kwargs
            )

class FractalMAR_huge_in256(FractalGen):
    def __init__(self, *args, **kwargs):
        super().__init__(
            img_size_list=(256, 16, 4, 1),
            embed_dim_list=(1280, 640, 320, 64),
            num_blocks_list=(40, 10, 5, 1),
            num_heads_list=(16, 8, 4, 4),
            generator_type_list=("mar", "mar", "mar", "ar"),
            fractal_level=0,
            **kwargs
            )