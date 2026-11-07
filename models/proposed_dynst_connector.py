"""
Dynamic Spatio-Temporal Difference Connector (DynSTConnector)
PROPOSED NOVELTY for Sign Language Translation with Small VLMs.

Key Contributions:
1. Motion-Difference Temporal Stream (MD-TS): Computes discrete temporal velocity
   and models gesture kinematics via temporal depthwise convolutions + dynamic gating.
2. Articulator-Aware Dynamic Spatial Pooling (ADSP): Uses patch-level motion variance
   to dynamically identify active sign articulators (hands, face) and filter out static background.
3. Lightweight Modality Projection: Compresses redundant visual tokens while preserving
   fine-grained spatio-temporal sign structures.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MotionDifferenceTemporalStream(nn.Module):
    """
    Computes first-order temporal feature differences and models gesture strokes:
    Δ_t = LayerNorm(H_t - H_{t-1})
    M_t = TemporalDepthwiseConv1D(Δ_t)
    g_t = Sigmoid(MLP([Pool(H_t) || Pool(Δ_t)]))
    H_tilde_t = H_t + g_t * W_m(M_t)
    """
    def __init__(self, d_model=1152, kernel_size=3, gating_reduction=4):
        super().__init__()
        self.d_model = d_model
        self.diff_norm = nn.LayerNorm(d_model)

        padding = kernel_size // 2
        # Temporal Depthwise-Separable 1D Convolution over time
        self.depthwise_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=padding,
            groups=d_model
        )
        self.pointwise_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=1
        )
        self.act = nn.SiLU()

        # Dynamic motion gating module
        gate_hidden = max(16, d_model // gating_reduction)
        self.gate_mlp = nn.Sequential(
            nn.Linear(d_model * 2, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, d_model),
            nn.Sigmoid()
        )
        self.proj_motion = nn.Linear(d_model, d_model)

    def forward(self, H):
        """
        Args:
            H: [B, T, P, D] spatial patch tokens from frozen encoder
        Returns:
            H_tilde: [B, T, P, D] motion-enhanced visual tokens
            delta: [B, T, P, D] raw temporal difference features
        """
        B, T, P, D = H.shape

        # Compute temporal velocity: Δ_t = H_t - H_{t-1}
        delta = torch.zeros_like(H)
        if T > 1:
            delta[:, 1:] = H[:, 1:] - H[:, :-1]
        delta = self.diff_norm(delta)

        # Reshape for 1D convolution over temporal dimension T
        # [B, T, P, D] -> [B * P, D, T]
        delta_perm = delta.permute(0, 2, 3, 1).contiguous().view(B * P, D, T)
        motion_conv = self.depthwise_conv(delta_perm)
        motion_conv = self.act(motion_conv)
        motion_conv = self.pointwise_conv(motion_conv)
        # Reshape back to [B, T, P, D]
        motion_feat = motion_conv.view(B, P, D, T).permute(0, 3, 1, 2).contiguous()

        # Compute global spatial pools for gating: [B, T, D]
        h_pool = H.mean(dim=2)
        delta_pool = delta.mean(dim=2)
        gate_input = torch.cat([h_pool, delta_pool], dim=-1) # [B, T, 2*D]
        gate = self.gate_mlp(gate_input).unsqueeze(2) # [B, T, 1, D]

        # Injected motion features
        motion_projected = self.proj_motion(motion_feat)
        H_tilde = H + gate * motion_projected

        return H_tilde, delta

class ArticulatorAwareSpatialPooling(nn.Module):
    """
    Computes inter-frame temporal variance per spatial patch:
    s_p = (1 / (T-1)) * sum_{t=2}^T || H_{t,p} - H_{t-1,p} ||_2
    Dynamically concentrates visual representation onto high-motion articulators
    (hands, face, mouth) and suppresses static background.
    """
    def __init__(self, d_model=1152, num_queries=16, saliency_beta=0.3):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries
        self.saliency_beta = saliency_beta

        # Learnable articulator query tokens
        self.query_embed = nn.Parameter(torch.randn(1, num_queries, d_model) * 0.02)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, H, delta):
        """
        Args:
            H: [B, T, P, D] motion-enhanced tokens
            delta: [B, T, P, D] temporal difference features
        Returns:
            pooled_tokens: [B, T, num_queries, D] concentrated articulator tokens
            saliency_scores: [B, P] patch-level motion variance
        """
        B, T, P, D = H.shape

        # Patch-level motion variance across time: s_p = mean_t(||delta_{t,p}||_2)
        # delta is [B, T, P, D]
        patch_saliency = torch.norm(delta, p=2, dim=-1).mean(dim=1) # [B, P]
        # Normalize saliency to [0, 1]
        s_min = patch_saliency.min(dim=1, keepdim=True)[0]
        s_max = patch_saliency.max(dim=1, keepdim=True)[0]
        s_norm = (patch_saliency - s_min) / (s_max - s_min + 1e-6) # [B, P]

        # Saliency bias: log(s_norm + 1e-4)
        saliency_bias = self.saliency_beta * torch.log(s_norm + 1e-4) # [B, P]

        # Per-frame cross-attention from learnable articulator queries to patches
        # Query: [B * T, K, D], Key/Value: [B * T, P, D]
        queries = self.query_embed.repeat(B * T, 1, 1)
        H_flat = H.view(B * T, P, D)
        keys = self.k_proj(H_flat)
        values = self.v_proj(H_flat)

        attn_scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(D) # [B*T, K, P]

        # Broadcast saliency bias across time and query dimensions
        # saliency_bias: [B, P] -> repeat_interleave across T -> [B*T, 1, P]
        bias_flat = saliency_bias.repeat_interleave(T, dim=0).unsqueeze(1)
        attn_scores = attn_scores + bias_flat

        attn_weights = F.softmax(attn_scores, dim=-1)
        pooled = torch.bmm(attn_weights, values) # [B*T, K, D]
        pooled = self.out_proj(pooled)
        pooled = self.norm(pooled)

        pooled_tokens = pooled.view(B, T, self.num_queries, D)
        return pooled_tokens, s_norm

class DynSTConnector(nn.Module):
    """
    Unified Dynamic Spatio-Temporal Difference Connector:
    Frozen Vision Tokens -> Motion-Difference Stream -> Articulator Pooling -> LLM Projection
    """
    def __init__(
        self,
        vision_dim=1152,
        llm_dim=2048,
        hidden_dim=2048,
        num_articulator_queries=16,
        saliency_beta=0.3,
        dropout=0.1
    ):
        super().__init__()
        self.vision_dim = vision_dim
        self.llm_dim = llm_dim
        self.num_articulator_queries = num_articulator_queries

        # Component 1: Motion Difference Temporal Stream
        self.motion_stream = MotionDifferenceTemporalStream(d_model=vision_dim)

        # Component 2: Articulator-Aware Dynamic Spatial Pooling
        self.articulator_pool = ArticulatorAwareSpatialPooling(
            d_model=vision_dim,
            num_queries=num_articulator_queries,
            saliency_beta=saliency_beta
        )

        # Component 3: Modality Projector to LLM Space
        self.projector = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, llm_dim),
            nn.LayerNorm(llm_dim)
        )

    def forward(self, spatial_tokens):
        """
        Args:
            spatial_tokens: [B, T, P, D_v]
        Returns:
            visual_tokens: [B, T * num_queries, D_llm]
            meta: dictionary with motion features and saliency map for analysis
        """
        B, T, P, D = spatial_tokens.shape

        # Step 1: Inject temporal motion dynamics
        H_tilde, delta = self.motion_stream(spatial_tokens)

        # Step 2: Pool articulators based on motion variance
        pooled_articulators, saliency = self.articulator_pool(H_tilde, delta) # [B, T, K, D_v]

        # Step 3: Project to LLM embedding dimension
        K = self.num_articulator_queries
        flat_articulators = pooled_articulators.view(B, T * K, D)
        visual_tokens = self.projector(flat_articulators) # [B, T * K, D_llm]

        # Global pooled visual embedding for contrastive anchoring: [B, D_llm]
        pooled_video_repr = visual_tokens.mean(dim=1)

        return visual_tokens, {
            "pooled_video_repr": pooled_video_repr,
            "saliency": saliency,
            "temporal_delta_norm": torch.norm(delta, dim=-1).mean()
        }

