# Re-exports from the shared pos_embed module.
# Kept for backward compatibility; prefer importing from medlat.modules.pos_embed directly.
from medlat.modules.pos_embed import (
    get_1d_sincos_pos_embed_from_grid,
    get_1d_sincos_pos_embed,
    get_2d_sincos_pos_embed_from_grid,
    get_2d_sincos_pos_embed,
    get_3d_sincos_pos_embed_from_grid,
    get_3d_sincos_pos_embed,
    get_4d_sincos_pos_embed_from_grid,
    get_4d_sincos_pos_embed,
    interpolate_pos_embed,
)
