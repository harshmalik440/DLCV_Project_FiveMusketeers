"""
Loss functions for Sign Language Translation:
1. Standard Autoregressive Cross-Entropy Loss (with padding mask via ignore_index)
2. InfoNCE Contrastive Semantic Anchoring Loss with memory queue for B=1 support
3. DynST Composite Loss combining generation and visual-semantic grounding

Fix B (audit): When physical batch_size=1, standard InfoNCE collapses to a trivial
loss. We maintain a sliding memory queue of recent (video, text) embeddings to
supply in-batch negatives even when the physical batch size is 1. Queue size K
defaults to 64 entries.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveSemanticLoss(nn.Module):
    """
    Symmetric InfoNCE loss with a momentum memory queue.

    When the physical batch size B >= 2, standard in-batch negatives are used.
    When B == 1, negatives are drawn from a FIFO queue of recent embeddings,
    so the true InfoNCE formulation is always active during training.
    """
    def __init__(self, temperature=0.07, queue_size=64, embed_dim=512):
        super().__init__()
        self.temperature = temperature
        self.queue_size = queue_size

        # FIFO queues for video and text embeddings
        self.register_buffer("video_queue", torch.randn(queue_size, embed_dim))
        self.register_buffer("text_queue", torch.randn(queue_size, embed_dim))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        # Normalize initial queue entries
        self.video_queue = F.normalize(self.video_queue, dim=-1)
        self.text_queue = F.normalize(self.text_queue, dim=-1)

    @torch.no_grad()
    def _enqueue(self, video_embeds, text_embeds):
        """Push current batch embeddings into the FIFO queue."""
        batch_size = video_embeds.size(0)
        ptr = int(self.queue_ptr.item())

        # Handle wrap-around
        if ptr + batch_size <= self.queue_size:
            self.video_queue[ptr:ptr + batch_size] = video_embeds.detach()
            self.text_queue[ptr:ptr + batch_size] = text_embeds.detach()
        else:
            overflow = (ptr + batch_size) - self.queue_size
            self.video_queue[ptr:] = video_embeds[:self.queue_size - ptr].detach()
            self.text_queue[ptr:] = text_embeds[:self.queue_size - ptr].detach()
            self.video_queue[:overflow] = video_embeds[self.queue_size - ptr:].detach()
            self.text_queue[:overflow] = text_embeds[self.queue_size - ptr:].detach()

        self.queue_ptr[0] = (ptr + batch_size) % self.queue_size

    def forward(self, video_embeds, text_embeds):
        """
        Args:
            video_embeds: [B, D] pooled visual representation
            text_embeds:  [B, D] pooled target text representation
        Returns:
            scalar symmetric InfoNCE loss
        """
        B, D = video_embeds.shape
        # Dynamically adjust queue dimension if embedding dimension differs
        if self.video_queue.size(-1) != D:
            self.video_queue = F.normalize(torch.randn(self.queue_size, D, device=video_embeds.device), dim=-1)
            self.text_queue = F.normalize(torch.randn(self.queue_size, D, device=text_embeds.device), dim=-1)
            self.queue_ptr.zero_()

        v_norm = F.normalize(video_embeds, p=2, dim=-1)
        t_norm = F.normalize(text_embeds, p=2, dim=-1)

        if B >= 2:
            # Standard in-batch InfoNCE
            sim_matrix = torch.matmul(v_norm, t_norm.t()) / self.temperature
            labels = torch.arange(B, device=video_embeds.device)
            loss_v2t = F.cross_entropy(sim_matrix, labels)
            loss_t2v = F.cross_entropy(sim_matrix.t(), labels)
        else:
            # B == 1: use memory queue for negatives
            # Positive pair: sim(v, t) for the current sample
            # Negatives: sim(v, t_queue) and sim(t, v_queue)
            # video->text: [1, 1+K] where first column is positive
            v2t_pos = torch.matmul(v_norm, t_norm.t()) / self.temperature  # [1, 1]
            v2t_neg = torch.matmul(v_norm, self.text_queue.t()) / self.temperature  # [1, K]
            v2t_logits = torch.cat([v2t_pos, v2t_neg], dim=1)  # [1, 1+K]

            t2v_pos = torch.matmul(t_norm, v_norm.t()) / self.temperature  # [1, 1]
            t2v_neg = torch.matmul(t_norm, self.video_queue.t()) / self.temperature  # [1, K]
            t2v_logits = torch.cat([t2v_pos, t2v_neg], dim=1)  # [1, 1+K]

            # Label 0 = positive is at index 0
            labels = torch.zeros(1, dtype=torch.long, device=video_embeds.device)
            loss_v2t = F.cross_entropy(v2t_logits, labels)
            loss_t2v = F.cross_entropy(t2v_logits, labels)

        # Enqueue current batch
        self._enqueue(v_norm, t_norm)

        return 0.5 * (loss_v2t + loss_t2v)


class DynSTLoss(nn.Module):
    """
    Combined objective: L_total = L_CE + lambda * L_CSA

    Fixes applied:
    - ignore_index=0 (pad token id) so padding tokens are excluded from CE loss.
    - InfoNCE always uses true contrastive formulation via memory queue.
    """
    def __init__(self, contrastive_weight=0.25, temperature=0.07,
                 label_smoothing=0.05, ignore_index=0,
                 queue_size=64, contrastive_dim=512):
        super().__init__()
        self.contrastive_weight = contrastive_weight
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing
        )
        if contrastive_weight > 0:
            self.contrastive_loss = ContrastiveSemanticLoss(
                temperature=temperature,
                queue_size=queue_size,
                embed_dim=contrastive_dim
            )
        else:
            self.contrastive_loss = None

    def forward(self, logits, labels, video_repr=None, text_repr=None):
        """
        Args:
            logits: [B, Seq_len, Vocab_size]
            labels: [B, Seq_len]
            video_repr: [B, D] optional video feature embedding
            text_repr: [B, D] optional text semantic embedding
        """
        vocab_size = logits.size(-1)
        flat_logits = logits.view(-1, vocab_size)
        flat_labels = labels.view(-1)

        loss_ce = self.ce_loss(flat_logits, flat_labels)
        loss_dict = {"loss_ce": loss_ce.item()}

        if (self.contrastive_loss is not None and
                video_repr is not None and text_repr is not None):
            loss_csa = self.contrastive_loss(video_repr, text_repr)
            loss_total = loss_ce + self.contrastive_weight * loss_csa
            loss_dict["loss_csa"] = loss_csa.item()
        else:
            loss_total = loss_ce
            loss_dict["loss_csa"] = 0.0

        loss_dict["loss_total"] = loss_total.item()
        return loss_total, loss_dict
