import math
import torch
from typing import Dict, Tuple, Optional, Union, List
from .wrapper import BaseWrapperModel
from .vqgan import ConvVQModel
from .maskbit import LFQBert
from src.losses.loss import MLMLoss
from .util.factorization import split_factorized_tokens
from .util.masking import get_mask_tokens
from medlat.modules.pos_embed import get_3d_sincos_pos_embed
from einops import rearrange


class Pseudo3DLFQBert(LFQBert):
    def __init__(
        self,
        img_size: int = 128,
        hidden_dim: int = 768,
        codebook_size: int = 1024,
        codebook_splits: int = 1,
        depth: int = 24,
        heads: int = 8,
        mlp_dim: int = 3072,
        dropout: float = 0.1,
        nclass: int = 1,
        input_stride: int = 16,
        use_prenorm: bool = False,
        num_time_steps: int = 32
    ):
        """
            num_time_steps: Number of time steps in the 3D sequence
        """
        # Calculate spatial size and sequence length
        spatial_size = img_size // input_stride
        seq_len = num_time_steps * spatial_size * spatial_size
        
        super().__init__(
            img_size=img_size,
            hidden_dim=hidden_dim,
            codebook_size=codebook_size,
            codebook_splits=codebook_splits,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            nclass=nclass,
            input_stride=input_stride,
            use_prenorm=use_prenorm
        )
        
        self.num_time_steps = num_time_steps
        self.spatial_size = spatial_size
        self.seq_len = seq_len
        
        # Generate 3D sin-cos positional embedding (not learnable)
        pos_embed_np = get_3d_sincos_pos_embed(
            embed_dim=hidden_dim,
            grid_depth=num_time_steps,
            grid_height=spatial_size,
            grid_width=spatial_size
        )  # shape: [seq_len, hidden_dim]
        self.register_buffer('pos_embed_3d', torch.from_numpy(pos_embed_np).float(), persistent=False)

    def forward(
        self,
        img_tokens: torch.Tensor,
        class_labels: torch.Tensor,
        drop_label_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:

        b = img_tokens.size(0)

        img_tokens = self.preprocess_tokens(img_tokens)

        cls_token = class_labels.view(b, -1)
        cls_token[drop_label_mask] = self.drop_label  # Drop condition
        cls_embedding = self.class_emb(cls_token.view(b, -1))

        projected_bit_tokens = self.input_proj(img_tokens)
        projected_bit_tokens = torch.cat([projected_bit_tokens, cls_embedding], dim=1)

        # Use 3D positional embedding
        pos_embeddings = self.pos_embed_3d
        x = projected_bit_tokens + pos_embeddings

        # transformer forward pass
        x = self.first_layer(x)
        x, attn = self.transformer(x, return_attn=return_attn)
        if self.use_prenorm:
            x = self.norm_after_transformer(x)
        x = self.last_layer(x)

        logits = rearrange(self.prediction_layer(x), "b n (m c) -> b n m c", c=self.effective_codebook_size, m=self.splits)
        logits = logits[:, :self.seq_len, ...]

        if return_attn:  # return list of attention
            return logits, attn

        return logits


class Pseudo3DWrapper(BaseWrapperModel):
    def __init__(
        self,
        mlm: Pseudo3DLFQBert,
        vq: ConvVQModel,
        loss: MLMLoss,
        train_mask_schedule_strategy: str = "arccos"
    ):
        super().__init__()
        
        self.train_mask_schedule_strategy = train_mask_schedule_strategy
        self.mlm = mlm
        self.vq = vq
        self.loss = loss
        self.patch_size = int(math.sqrt(self.mlm.spatial_size * self.mlm.spatial_size))
        
        self.lock_n_load()
    
    def lock_n_load(self):
        self.vq.eval()
        for p in self.vq.parameters():
            p.requires_grad = False
    
    def vq_encode(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            B, T, _, _, _ = x.shape
            encoded_tokens = []
            
            for t in range(T):
                x_t = x[:, t]  # [B, C, H, W]
                _, _ , (_, _, min_encoding_indices) = self.vq.encode(x_t)
                tokens_t = min_encoding_indices
                tokens_t = tokens_t.reshape(B, -1)
                encoded_tokens.append(tokens_t)
            
            encoded_tokens = torch.cat(encoded_tokens, dim=1)  # [B, T*seq_len]
            return encoded_tokens
    
    def vq_decode(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        seq_len = self.patch_size * self.patch_size
        T = x.shape[1] // seq_len
        
        x = x.reshape(B, T, seq_len)
        
        decoded_images = []
        z_channels = self.vq.quantize.embedding.weight.shape[1]
        for t in range(T):
            x_t = x[:, t]  # [B, seq_len]
            img_t = self.vq.decode_code(x_t, out_shape=(B, self.patch_size, self.patch_size, z_channels))
            decoded_images.append(img_t)
        
        decoded_images = torch.stack(decoded_images, dim=1)  # [B, T, C, H, W]
        return decoded_images
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        input_tokens = self.vq_encode(x) # [B, T*seq_len]
        
        input_tokens = split_factorized_tokens(
            input_tokens, 
            codebook_size=self.mlm.effective_codebook_size,
            splits=self.mlm.splits
        ) # [B, T, seq_len]
        
        masked_tokens, masks = get_mask_tokens(
            input_tokens,
            self.mlm.mask_token,
            mode=self.train_mask_schedule_strategy
        )
        
        class_tokens = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        drop_label_mask = torch.ones_like(class_tokens, dtype=torch.bool, device=x.device)
        
        logits = self.mlm(masked_tokens, class_tokens, drop_label_mask)
        
        loss, loss_dict = self.loss(logits, input_tokens, masks)
        
        return loss, loss_dict
