"""
Unit tests for Baseline and Proposed DynST-VLM architectures.
Verifies:
1. Frozen vision encoder gradients.
2. Tensor shape transformations through DynST connector.
3. Articulator saliency computation and motion difference stream.
4. Loss computation and backward gradient flow.
5. Autoregressive greedy decoding.
"""

import unittest
import torch

from models.vlm_baseline import BaselineVLMForSLT, StaticMLPConnector
from models.proposed_dynst_connector import DynSTConnector, MotionDifferenceTemporalStream, ArticulatorAwareSpatialPooling
from models.vlm_proposed import DynSTVLMForSLT
from models.losses import DynSTLoss, ContrastiveSemanticLoss

class TestModelArchitectures(unittest.TestCase):
    def setUp(self):
        self.B = 2
        self.T = 8
        self.C = 3
        self.H = 64
        self.W = 64
        self.patch_size = 16
        self.vision_dim = 128
        self.llm_dim = 256
        self.vocab_size = 50
        self.seq_len = 10

        self.frames = torch.randn(self.B, self.T, self.C, self.H, self.W)
        self.text_ids = torch.randint(0, self.vocab_size, (self.B, self.seq_len))

    def test_baseline_architecture(self):
        model = BaselineVLMForSLT(
            vision_dim=self.vision_dim,
            llm_dim=self.llm_dim,
            vocab_size=self.vocab_size,
            img_size=self.H,
            patch_size=self.patch_size,
            freeze_vision=True
        )

        # Check vision encoder is frozen
        for p in model.vision_encoder.parameters():
            self.assertFalse(p.requires_grad)

        # Forward pass
        out = model(self.frames, self.text_ids)
        self.assertIn("logits", out)
        self.assertEqual(out["logits"].shape, (self.B, self.seq_len, self.vocab_size))

        # Check decoding
        gen = model.generate(self.frames, max_len=5)
        self.assertEqual(gen.size(0), self.B)
        self.assertTrue(gen.size(1) <= 6)

    def test_dynst_connector(self):
        connector = DynSTConnector(
            vision_dim=self.vision_dim,
            llm_dim=self.llm_dim,
            hidden_dim=self.llm_dim,
            num_articulator_queries=8,
            saliency_beta=0.3
        )
        P = (self.H // self.patch_size) ** 2
        dummy_spatial = torch.randn(self.B, self.T, P, self.vision_dim)

        out_tokens, meta = connector(dummy_spatial)
        # Expected visual tokens: [B, T * K, llm_dim] where K=8
        self.assertEqual(out_tokens.shape, (self.B, self.T * 8, self.llm_dim))
        self.assertIn("pooled_video_repr", meta)
        self.assertEqual(meta["pooled_video_repr"].shape, (self.B, self.llm_dim))
        self.assertIn("saliency", meta)
        self.assertEqual(meta["saliency"].shape, (self.B, P))

    def test_proposed_dynst_vlm_forward_and_loss(self):
        model = DynSTVLMForSLT(
            vision_dim=self.vision_dim,
            llm_dim=self.llm_dim,
            vocab_size=self.vocab_size,
            img_size=self.H,
            patch_size=self.patch_size,
            num_articulator_queries=8,
            contrastive_dim=64,
            freeze_vision=True
        )

        out = model(self.frames, self.text_ids)
        self.assertIn("logits", out)
        self.assertIn("video_proj", out)
        self.assertIn("text_proj", out)

        self.assertEqual(out["logits"].shape, (self.B, self.seq_len, self.vocab_size))
        self.assertEqual(out["video_proj"].shape, (self.B, 64))
        self.assertEqual(out["text_proj"].shape, (self.B, 64))

        criterion = DynSTLoss(contrastive_weight=0.25, temperature=0.07)
        labels = self.text_ids
        loss, loss_dict = criterion(out["logits"], labels, out["video_proj"], out["text_proj"])

        self.assertTrue(loss.item() > 0)
        self.assertIn("loss_ce", loss_dict)
        self.assertIn("loss_csa", loss_dict)

        # Check backward pass
        loss.backward()

        # Connector parameters should receive gradients
        has_grad = False
        for p in model.connector.parameters():
            if p.grad is not None and torch.norm(p.grad) > 0:
                has_grad = True
                break
        self.assertTrue(has_grad, "Connector parameters did not receive gradients")

if __name__ == "__main__":
    unittest.main()

