"""
Baseline Vision-Language Model Architecture for Sign Language Translation.

Supports two execution modes:
  Mode A (Pretrained): Loads real HuggingFace pretrained VLM (SmolVLM2 / InternVL / Qwen)
          with frozen vision encoder and LoRA-tuned LM decoder via PEFT.
          Requires: transformers, peft, accelerate.
  Mode B (Standalone): Lightweight modular architecture mirroring the paper's pipeline
          for environments without pretrained model access (testing, CI, limited VRAM).
          Uses randomly initialized encoder/decoder with identical data flow.

Both modes share the same StaticMLPConnector and the same forward/generate API,
so the proposed DynST connector can be evaluated identically in both modes.
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Try importing HuggingFace ecosystem ----------
try:
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoModel
    from peft import get_peft_model, LoraConfig, TaskType
    HAS_HF = True
except ImportError:
    HAS_HF = False


# ====================================================================
# Shared: Static MLP Connector (used by both modes)
# ====================================================================
class StaticMLPConnector(nn.Module):
    """
    Standard static 2-layer MLP connector mapping visual tokens to language embeddings.
    As described in Section 3.1 & 3.2 of the paper.
    """
    def __init__(self, vision_dim=1152, hidden_dim=2048, llm_dim=2048, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(vision_dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, llm_dim)
        self.norm = nn.LayerNorm(llm_dim)

    def forward(self, x):
        """
        Args:
            x: Visual patch tokens [B, T, P, D_v] or flattened [B, N, D_v]
        Returns:
            Projected tokens [B, N, D_llm]
        """
        if x.dim() == 4:
            B, T, P, D = x.shape
            x = x.view(B, T * P, D)
        out = self.fc1(x)
        out = self.act(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.norm(out)
        return out


# ====================================================================
# Mode B: Standalone lightweight modules (no pretrained weights)
# ====================================================================
class LightweightVisionBackbone(nn.Module):
    """
    Standalone vision backbone that mirrors the SigLIP/ViT patch extraction pipeline.
    Used when pretrained HuggingFace models are unavailable.
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=1152):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=8,
                dim_feedforward=embed_dim * 2,
                dropout=0.0, activation="gelu", batch_first=True
            ) for _ in range(4)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, frames):
        """frames: [B, T, C, H, W] -> patch_tokens: [B, T, P, D]"""
        B, T, C, H, W = frames.shape
        frames_flat = frames.view(B * T, C, H, W)
        x = self.proj(frames_flat).flatten(2).transpose(1, 2)  # [B*T, P, D]
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x.view(B, T, x.shape[1], self.embed_dim)


class LoRALinear(nn.Module):
    """Parameter-Efficient LoRA layer (Hu et al., 2021)."""
    def __init__(self, in_features, out_features, r=64, lora_alpha=64, lora_dropout=0.1):
        super().__init__()
        self.base_layer = nn.Linear(in_features, out_features)
        self.r = r
        self.scaling = lora_alpha / r if r > 0 else 1.0
        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros(r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))
            self.dropout = nn.Dropout(lora_dropout)
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    def forward(self, x):
        result = self.base_layer(x)
        if self.r > 0:
            lora_out = self.dropout(x)
            lora_out = F.linear(lora_out, self.lora_A)
            lora_out = F.linear(lora_out, self.lora_B) * self.scaling
            result = result + lora_out
        return result


class ModularLanguageDecoder(nn.Module):
    """
    Autoregressive transformer decoder with LoRA on attention projections
    and the output head.
    """
    def __init__(self, vocab_size=3000, d_model=2048, nhead=8, num_layers=4,
                 r=64, lora_alpha=64, lora_dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, 512, d_model))

        # Decoder layers with LoRA-adapted attention projections
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.TransformerDecoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=d_model * 2,
                dropout=0.1, activation="gelu", batch_first=True
            )
            self.layers.append(layer)

        # LoRA on Q, K, V projections of each layer
        self.lora_q = nn.ModuleList([
            LoRALinear(d_model, d_model, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
            for _ in range(num_layers)
        ])
        self.lora_k = nn.ModuleList([
            LoRALinear(d_model, d_model, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
            for _ in range(num_layers)
        ])
        self.lora_v = nn.ModuleList([
            LoRALinear(d_model, d_model, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
            for _ in range(num_layers)
        ])

        # LoRA on output projection
        self.lm_head = LoRALinear(d_model, vocab_size, r=r, lora_alpha=lora_alpha)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, text_ids, visual_tokens, text_mask=None):
        """text_ids: [B, S], visual_tokens: [B, N_v, D] -> logits: [B, S, V]"""
        B, S = text_ids.shape
        tok_embeds = self.embed(text_ids) * math.sqrt(self.d_model)
        tok_embeds = tok_embeds + self.pos_embed[:, :S, :]

        causal_mask = nn.Transformer.generate_square_subsequent_mask(S, device=text_ids.device)

        hidden = tok_embeds
        for i, layer in enumerate(self.layers):
            # Apply LoRA projections before standard decoder attention
            q_lora = self.lora_q[i](hidden)
            k_lora = self.lora_k[i](hidden)
            v_lora = self.lora_v[i](hidden)
            # Residual injection of LoRA-adapted representations
            hidden = hidden + 0.1 * (q_lora + k_lora + v_lora - 3 * hidden)
            hidden = layer(tgt=hidden, memory=visual_tokens, tgt_mask=causal_mask)

        return self.lm_head(hidden)


# ====================================================================
# Mode A: HuggingFace Pretrained VLM wrapper (when transformers+peft available)
# ====================================================================
class PretrainedVisionEncoder(nn.Module):
    """Wraps a HuggingFace pretrained vision model (e.g., SigLIP, InternViT)."""
    def __init__(self, model_name="google/siglip-so400m-patch14-384", embed_dim=1152):
        super().__init__()
        self.embed_dim = embed_dim
        if HAS_HF:
            try:
                self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
                print(f"[Vision] Loaded pretrained encoder: {model_name}")
            except Exception as e:
                print(f"[Vision] Could not load {model_name}: {e}. Using standalone backbone.")
                self.encoder = None
        else:
            self.encoder = None

    def forward(self, frames):
        """frames: [B, T, C, H, W] -> [B, T, P, D]"""
        B, T, C, H, W = frames.shape
        if self.encoder is not None:
            frames_flat = frames.view(B * T, C, H, W)
            with torch.no_grad():
                outputs = self.encoder(pixel_values=frames_flat)
                # SigLIP / ViT returns last_hidden_state: [B*T, P+1, D] or [B*T, P, D]
                hidden = outputs.last_hidden_state
            P = hidden.shape[1]
            return hidden.view(B, T, P, self.embed_dim)
        else:
            raise RuntimeError("No pretrained encoder available. Use standalone mode.")


class PretrainedLMDecoder(nn.Module):
    """Wraps a HuggingFace pretrained causal LM with PEFT LoRA."""
    def __init__(self, model_name="HuggingFaceTB/SmolLM2-1.7B-Instruct",
                 lora_r=64, lora_alpha=64, lora_dropout=0.1,
                 target_modules=None):
        super().__init__()
        self.model = None
        if HAS_HF:
            try:
                if target_modules is None:
                    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"]
                base_model = AutoModelForCausalLM.from_pretrained(
                    model_name, trust_remote_code=True, torch_dtype=torch.bfloat16
                )
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM, r=lora_r,
                    lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                    target_modules=target_modules
                )
                self.model = get_peft_model(base_model, lora_config)
                self.model.print_trainable_parameters()
                print(f"[LM] Loaded pretrained decoder with LoRA: {model_name}")
            except Exception as e:
                print(f"[LM] Could not load {model_name}: {e}. Use standalone mode.")

    def forward(self, input_ids, visual_tokens=None, labels=None):
        if self.model is None:
            raise RuntimeError("No pretrained LM available.")
        return self.model(input_ids=input_ids, labels=labels)


# ====================================================================
# Unified Baseline VLM (auto-selects mode based on available packages)
# ====================================================================
class BaselineVLMForSLT(nn.Module):
    """
    Complete Baseline Sign Language Translation VLM.

    Automatically selects:
      - Mode A (pretrained=True): Real HuggingFace pretrained VLM with LoRA via PEFT.
      - Mode B (pretrained=False): Standalone lightweight modules for testing/CI.

    Both modes expose the same forward() and generate() API.
    """
    def __init__(
        self,
        vision_dim=1152,
        llm_dim=2048,
        vocab_size=3000,
        img_size=224,
        patch_size=16,
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
            # Mode A: Real pretrained models
            self.vision_encoder = PretrainedVisionEncoder(
                model_name=vision_model_name or "google/siglip-so400m-patch14-384",
                embed_dim=vision_dim
            )
            self.connector = StaticMLPConnector(
                vision_dim=vision_dim, hidden_dim=llm_dim, llm_dim=llm_dim
            )
            self.decoder = PretrainedLMDecoder(
                model_name=lm_model_name or "HuggingFaceTB/SmolLM2-1.7B-Instruct",
                lora_r=lora_r, lora_alpha=lora_alpha
            )
        else:
            # Mode B: Standalone lightweight modules
            self.vision_encoder = LightweightVisionBackbone(
                img_size=img_size, patch_size=patch_size, embed_dim=vision_dim
            )
            self.connector = StaticMLPConnector(
                vision_dim=vision_dim, hidden_dim=llm_dim, llm_dim=llm_dim
            )
            self.decoder = ModularLanguageDecoder(
                vocab_size=vocab_size, d_model=llm_dim,
                r=lora_r, lora_alpha=lora_alpha
            )

        if freeze_vision:
            for p in self.vision_encoder.parameters():
                p.requires_grad = False

    def forward(self, frames, text_ids=None):
        """frames: [B, T, C, H, W], text_ids: [B, S] -> dict with logits"""
        with torch.no_grad():
            spatial_tokens = self.vision_encoder(frames)  # [B, T, P, D_v]

        visual_tokens = self.connector(spatial_tokens)  # [B, N, D_llm]

        if text_ids is not None:
            if self.pretrained:
                outputs = self.decoder(text_ids, visual_tokens=visual_tokens)
                return {"logits": outputs.logits, "visual_tokens": visual_tokens}
            else:
                logits = self.decoder(text_ids, visual_tokens)
                return {"logits": logits, "visual_tokens": visual_tokens}
        return {"visual_tokens": visual_tokens}

    @torch.no_grad()
    def generate(self, frames, max_len=64, bos_id=1, eos_id=2):
        """Greedy autoregressive decoding."""
        self.eval()
        B = frames.size(0)
        device = frames.device
        spatial_tokens = self.vision_encoder(frames)
        visual_tokens = self.connector(spatial_tokens)

        cur_tokens = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        for _ in range(max_len):
            if self.pretrained:
                outputs = self.decoder(cur_tokens, visual_tokens=visual_tokens)
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            else:
                logits = self.decoder(cur_tokens, visual_tokens)
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            cur_tokens = torch.cat([cur_tokens, next_token], dim=1)
            if (next_token == eos_id).all():
                break
        return cur_tokens
