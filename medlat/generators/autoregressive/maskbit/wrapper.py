import math
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from abc import ABC
from accelerate import cpu_offload_with_hook
from huggingface_hub import hf_hub_download
import gc
from src.accelerator import AccelerateParent
from src.losses.loss import MLMLoss
from .ldm_vq import VQModel
from .util.factorization import split_factorized_tokens
from .util.masking import get_mask_tokens
from .vqgan import ConvVQModel
from .maskbit import LFQBert
from src.utils import instantiate_from_config


class BaseWrapperModel(ABC, nn.Module, AccelerateParent):
    def __init__(self) -> None:
        super(BaseWrapperModel, self).__init__()

    def load_pretrained(self, *args, **kwargs): 
        pass

    def lock_n_load(self, *args, **kwargs):
        pass

    def encode(self, *args, **kwargs):
        pass 

    def vq_encode(self, x, *args, **kwargs):
        return x

    def vq_decode(self, x, *args, **kwargs):
        return x

    def enable_cpu_offload(self, *args, **kwargs):
        pass

    def init_from_ckpt(self, path):
        sd = torch.load(path, map_location="cpu", weights_only=False)["model"]
        msg = self.load_state_dict(sd, strict=True)
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Loading pre-trained {self.__class__.__name__}")
        print("Missing keys:")
        print(msg.missing_keys)
        print("Unexpected keys:")
        print(msg.unexpected_keys)
        print(f"Restored from {path}")

class MaskBit(BaseWrapperModel):
    def __init__(
        self,
        mlm: LFQBert,
        vq: VQModel,
        loss: MLMLoss,
        train_mask_schedule_strategy = "arccos",
        ckpt_path: Optional[str] = None,
        use_pretrained_mlm: bool = False
    ):
        
        super().__init__()
        
        self.train_mask_schedule_strategy = train_mask_schedule_strategy
        self.mlm = mlm
        self.vq = vq
        self.loss = loss
        self.patch_size = int(math.sqrt(self.mlm.seq_len))

        # self.vq, self.vq_hook = cpu_offload_with_hook(self.vq, execution_device=self.device) # TODO: What is this hook?

        # TODO: What model checkpoint do we refer to here? The Wrapper or the individual models?
        if ckpt_path:
            self.init_from_ckpt(ckpt_path)

        if use_pretrained_mlm:
            self.init_mlm_model()
        
        self.lock_n_load()
    
    def lock_n_load(self):
        locks = [self.vq]
        for lock in locks:
            lock.eval()
            for p in lock.parameters():
                p.requires_grad = False

    
    def init_mlm_model(self):
        token_size = self.mlm.bits
        print(f"Token size: {token_size}")  
        repo_id =  f"markweber/maskbit_generator_{token_size}bit"
        filename = f"maskbit_generator_{token_size}bit.bin"
        mlm_model_path = hf_hub_download(repo_id=repo_id, filename=filename)

        print(f"Loading pretrained MLM of {token_size} bits from HuggingFace")
        print(f"Path: {mlm_model_path}")
        
        rename_dict = {"token_emb": "input_proj"}
        self.mlm.load_pretrained(mlm_model_path,rename_keys=rename_dict)



    def vq_encode(self, x):
        with torch.no_grad():
            if isinstance(self.vq, VQModel):
                _, _ , (_, _, min_encoding_indices) = self.vq.encode(x)
                input_tokens = min_encoding_indices 
            elif isinstance(self.vq, ConvVQModel):
                _, encoder_dict = self.vq.encode(x)
                input_tokens = encoder_dict["min_encoding_indices"]
            else:
                raise NotImplementedError(f"Encoding not implemented for {self.vq.__class__.__name__}")

            # Preserve batch dimension and flatten spatial dimensions
            input_tokens = input_tokens.reshape(x.shape[0], -1)
            return input_tokens

    def vq_decode(self, x):
        if isinstance(self.vq, VQModel):
            z_channels = self.vq.quantize.embedding.weight.shape[1]
            reconstructed_images = self.vq.decode_code(x, out_shape=(x.shape[0], self.patch_size, self.patch_size, z_channels))
        elif isinstance(self.vq, ConvVQModel):
            reconstructed_images = self.vq.decode_tokens(x)
        else:
            raise NotImplementedError(f"Decoding not implemented for {self.vq.__class__.__name__}")
        return reconstructed_images

    def forward(self, x, *args, **kwargs):
        input_tokens = self.vq_encode(x)
        input_tokens = split_factorized_tokens(input_tokens, codebook_size=self.mlm.effective_codebook_size, splits=self.mlm.splits)

        masked_tokens, masks = get_mask_tokens(
            input_tokens,
            self.mlm.mask_token,
            mode=self.train_mask_schedule_strategy
        )

        # Create class tokens on the same device as input
        class_tokens = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        drop_label_mask = torch.ones_like(class_tokens, dtype=torch.bool, device=x.device)
        logits = self.mlm(masked_tokens, class_tokens, drop_label_mask)
        maskgit_loss, loss_dict = self.loss(logits, input_tokens, masks)

        del class_tokens, drop_label_mask
        gc.collect()
        torch.cuda.empty_cache()

        return maskgit_loss, loss_dict