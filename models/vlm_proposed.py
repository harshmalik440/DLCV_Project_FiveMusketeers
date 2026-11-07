"""
Complete Proposed DynST-VLM Architecture for Sign Language Translation:
- Frozen Vision Encoder (SigLIP / ViT / Standalone)
- Dynamic Spatio-Temporal Difference Connector (DynSTConnector)
  * Motion-Difference Temporal Stream (MD-TS)
  * Motion-Guided Dynamic Spatial Pooling (MDSP)
- Contrastive Semantic Projection Head for Anchoring Loss (InfoNCE with FIFO queue)
- LoRA-tuned Autoregressive Language Model Decoder (Attention Q,K,V + Output projection)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vlm_baseline import (
    LightweightVisionBackbone,
    ModularLanguageDecoder,
    PretrainedVisionEncoder,
    PretrainedLMDecoder,
    HAS_HF
)
from .proposed_dynst_connector import DynSTConnector

class DynSTVLMForSLT(nn.Module):
    """
    Novel DynST-VLM Model:
    Combines frozen vision backbone + dynamic temporal difference connector +
    contrastive semantic anchoring + parameter-efficient LoRA language decoder.

    Supports both:
      - Mode A (pretrained=True): Pretrained SigLIP/InternViT + SmolLM2/LLaMA with LoRA
      - Mode B (pretrained=False): Standalone modular backbone for testing/CI
    """
    def __init__(
        self,
        vision_dim=1152,
        llm_dim=2048,
        vocab_size=3000,
        img_size=224,
        patch_size=16,
        num_articulator_queries=16,
        saliency_beta=0.3,
        contrastive_dim=512,
        lora_r=64,
        lora_alpha=64,
        freeze_vision=True,
        pretrained=False,
        vision_model_name=None,
        lm_model_name=None
    ):
        super().__init__()
        self.pretrained = pretrained and HAS_HF

        if self.pretrained:
            self.vision_encoder = PretrainedVisionEncoder(
                model_name=vision_model_name or "google/siglip-so400m-patch14-384",
                embed_dim=vision_dim
            )
            self.decoder = PretrainedLMDecoder(
                model_name=lm_model_name or "HuggingFaceTB/SmolLM2-1.7B-Instruct",
                lora_r=lora_r,
                lora_alpha=lora_alpha
            )
        else:
            self.vision_encoder = LightweightVisionBackbone(
                img_size=img_size,
                patch_size=patch_size,
                embed_dim=vision_dim
            )
            self.decoder = ModularLanguageDecoder(
                vocab_size=vocab_size,
                d_model=llm_dim,
                r=lora_r,
                lora_alpha=lora_alpha
            )

        self.connector = DynSTConnector(
            vision_dim=vision_dim,
            llm_dim=llm_dim,
            hidden_dim=llm_dim,
            num_articulator_queries=num_articulator_queries,
            saliency_beta=saliency_beta
        )

        # Contrastive Semantic Anchoring Projections
        self.video_contrastive_head = nn.Sequential(
            nn.Linear(llm_dim, contrastive_dim),
            nn.GELU(),
            nn.Linear(contrastive_dim, contrastive_dim)
        )
        self.text_contrastive_head = nn.Sequential(
            nn.Linear(llm_dim, contrastive_dim),
            nn.GELU(),
            nn.Linear(contrastive_dim, contrastive_dim)
        )

        if freeze_vision:
            for p in self.vision_encoder.parameters():
                p.requires_grad = False

    def forward(self, frames, text_ids=None):
        """
        Args:
            frames: [B, T, C, H, W]
            text_ids: [B, S] optional ground truth token ids
        Returns:
            Dictionary containing logits, visual_tokens, video_proj, text_proj, meta
        """
        with torch.no_grad():
            spatial_tokens = self.vision_encoder(frames) # [B, T, P, D_v]

        # DynST connector forward pass
        visual_tokens, meta = self.connector(spatial_tokens)

        # Contrastive video embedding
        pooled_v = meta["pooled_video_repr"]
        video_proj = self.video_contrastive_head(pooled_v)

        output = {
            "visual_tokens": visual_tokens,
            "video_proj": video_proj,
            "meta": meta
        }

        if text_ids is not None:
            if self.pretrained:
                lm_out = self.decoder(text_ids, visual_tokens=visual_tokens)
                output["logits"] = lm_out.logits
            else:
                logits = self.decoder(text_ids, visual_tokens)
                output["logits"] = logits

            # Extract target text representation for contrastive anchoring
            with torch.no_grad():
                if hasattr(self.decoder, "embed"):
                    text_embeds = self.decoder.embed(text_ids)
                else:
                    text_embeds = self.decoder.model.get_input_embeddings()(text_ids)
                text_pooled = text_embeds.mean(dim=1)
            text_proj = self.text_contrastive_head(text_pooled)
            output["text_proj"] = text_proj

        return output

    @torch.no_grad()
    def generate(self, frames, max_len=64, bos_id=1, eos_id=2):
        """
        Greedy autoregressive decoding for sign language translation.
        """
        self.eval()
        B = frames.size(0)
        device = frames.device

        spatial_tokens = self.vision_encoder(frames)
        visual_tokens, _ = self.connector(spatial_tokens)

        cur_tokens = torch.full((B, 1), bos_id, dtype=torch.long, device=device)

        for _ in range(max_len):
            if self.pretrained:
                lm_out = self.decoder(cur_tokens, visual_tokens=visual_tokens)
                next_token = torch.argmax(lm_out.logits[:, -1, :], dim=-1, keepdim=True)
            else:
                logits = self.decoder(cur_tokens, visual_tokens)
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            cur_tokens = torch.cat([cur_tokens, next_token], dim=1)
            if (next_token == eos_id).all():
                break

        return cur_tokens
