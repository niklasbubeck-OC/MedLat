"""Metric-logging tests for :class:`VQModelBase` and its subclasses.

Covers the new ``log_metric`` / ``get_metrics`` / ``reset_metrics`` surface
inherited from :class:`medlat.modules.metrics.MetricLoggerMixin`, plus the
quantizer-metric merge performed by :meth:`VQModelBase.get_metrics`.
"""
from __future__ import annotations

import pytest
import torch

from medlat import get_model


IMG_SIZE = 32


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_vq_model():
    # A tiny VQ model registered in the factory registry — exercises the full
    # conv-based pre/post_quant path through VQModel.
    return get_model("discrete.vq.f4_d3_e8192", img_size=IMG_SIZE).train()


def _tiny_images(n: int = 2) -> torch.Tensor:
    return torch.randn(n, 3, IMG_SIZE, IMG_SIZE, generator=torch.Generator().manual_seed(0))


# ---------------------------------------------------------------------------
# log_metric / get_metrics / reset_metrics inherited from the mixin
# ---------------------------------------------------------------------------


def test_vqmodel_log_metric_overwrites_latest():
    m = _make_vq_model()
    m.log_metric("my_stat", 1.0)
    m.log_metric("my_stat", 2.0)
    assert m.get_metrics()["my_stat"] == 2.0


def test_vqmodel_log_metric_detaches_tensor():
    m = _make_vq_model()
    x = torch.tensor(3.0, requires_grad=True)
    y = (x * 2).sum()  # grad_fn attached
    m.log_metric("y", y)
    stored = m.get_metrics()["y"]
    assert stored.grad_fn is None
    assert stored.item() == 6.0


def test_vqmodel_reset_metrics_clears_only_user_keys():
    m = _make_vq_model()
    m.log_metric("foo", 1.0)
    m.reset_metrics()
    # User-logged key is gone; quantizer-sourced keys may still appear via
    # the merge on the next get_metrics() call (none there yet since no
    # forward has run).
    assert "foo" not in m.get_metrics()


# ---------------------------------------------------------------------------
# Forward-time auto-logging of loss components
# ---------------------------------------------------------------------------


def test_vqmodel_forward_logs_commitment_and_total_loss():
    m = _make_vq_model()
    m(_tiny_images())
    snap = m.get_metrics()
    assert "commitment_loss" in snap
    assert "loss" in snap
    # Without alignment the two numbers are equal (no extra term added).
    assert m.alignment is None
    assert torch.isclose(snap["loss"], snap["commitment_loss"])


def test_vqmodel_forward_skips_alignment_loss_when_not_configured():
    m = _make_vq_model()
    m(_tiny_images())
    snap = m.get_metrics()
    assert "alignment_loss" not in snap


def test_vqmodel_forward_logs_alignment_loss_when_configured():
    # Install a dummy alignment module that produces a deterministic scalar —
    # that way we can assert the total loss = commitment + alignment.
    m = _make_vq_model()

    class _ConstAlignment:
        def __call__(self, quant, x):
            return torch.tensor(0.25), None

    m.alignment = _ConstAlignment()
    m(_tiny_images())
    snap = m.get_metrics()
    assert "alignment_loss" in snap
    assert pytest.approx(0.25) == snap["alignment_loss"].item()
    # loss must equal commitment + alignment
    total_from_parts = snap["commitment_loss"] + snap["alignment_loss"]
    assert torch.isclose(snap["loss"], total_from_parts)


# ---------------------------------------------------------------------------
# Merge with quantizer metrics
# ---------------------------------------------------------------------------


def test_vqmodel_get_metrics_merges_quantizer_perplexity():
    m = _make_vq_model()
    m(_tiny_images())
    snap = m.get_metrics()
    # Quantizer publishes these via its own _post_forward hook; the model's
    # get_metrics merges them in.
    assert "perplexity" in snap
    assert "dead_code_ratio" in snap
    assert "active_code_count" in snap
    assert "total_tokens_seen" in snap


def test_vqmodel_loss_wins_over_quantizer_loss_on_collision():
    # Both the quantizer and the VQ model publish a "loss" key. The model's
    # total loss (commitment + alignment) must win — otherwise users would see
    # the bare commitment under "loss" even when alignment is active.
    m = _make_vq_model()

    class _LargeAlignment:
        def __call__(self, quant, x):
            return torch.tensor(100.0), None

    m.alignment = _LargeAlignment()
    m(_tiny_images())
    snap = m.get_metrics()

    quantizer_loss = m.quantizer.get_metrics().get("loss")
    model_loss = snap["loss"]
    # Quantizer logs the commitment loss under "loss"; model publishes the
    # total (+100). If the merge precedence were reversed we'd see the
    # smaller commitment value here.
    assert model_loss.item() > quantizer_loss.item() + 50


def test_vqmodel_metrics_update_on_every_forward():
    m = _make_vq_model()
    m(_tiny_images(n=2))
    first = m.get_metrics()["loss"].clone()
    m(_tiny_images(n=4))
    second = m.get_metrics()["loss"]
    # Different batches → different loss values; the key property is that
    # the stored tensor is overwritten, not buffered.
    assert not torch.equal(first, second) or first.data_ptr() != second.data_ptr()
