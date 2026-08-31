# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the packed joint-QKV projection recipes and their dispatch
seam (``tensorrt_llm/_torch/visual_gen/modules/packed_qkv.py``).

The packed projection is the concat-elimination fast lane used by Qwen-Image
joint attention: the selected recipe's functional leaf op writes both
per-stream merged-QKV projections straight into row slices of one packed
buffer, replacing the per-stream projections + seq-dim ``torch.cat``.
"""

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
from tensorrt_llm._torch.visual_gen.models.qwen_image.transformer_qwen_image import (
    QwenJointAttention,
)

# Importing the module registers the trtllm_vgoa::packed_qkv_proj_* leaf ops.
from tensorrt_llm._torch.visual_gen.modules.packed_qkv import (
    build_packed_qkv,
    linear_supports_packed_addmm,
    select_packed_qkv_recipe,
)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

# Small stand-ins for the production shape (txt [1,1024,3072] +
# img [1,6889,3072] -> [1,7913,9216]); the packed dim stays a multiple of
# 128 like the real q_dim + 2*kv_dim.
HIDDEN = 128
PACKED = 3 * HIDDEN
S_TXT = 17
S_IMG = 64


class _QwenJointAttentionWithoutRope(QwenJointAttention):
    """Keep the production QKV preparation path while isolating its output."""

    def apply_packed_qk_norm_rope(self, *args: object, **kwargs: object) -> None:
        return None


def _make_inputs(device="cuda", dtype=torch.bfloat16):
    torch.manual_seed(0)
    txt = torch.randn(1, S_TXT, HIDDEN, device=device, dtype=dtype)
    img = torch.randn(1, S_IMG, HIDDEN, device=device, dtype=dtype)
    w_txt = torch.randn(PACKED, HIDDEN, device=device, dtype=dtype)
    b_txt = torch.randn(PACKED, device=device, dtype=dtype)
    w_img = torch.randn(PACKED, HIDDEN, device=device, dtype=dtype)
    b_img = torch.randn(PACKED, device=device, dtype=dtype)
    return txt, img, w_txt, b_txt, w_img, b_img


def _make_merged_linear(dtype=torch.bfloat16):
    linear = Linear(HIDDEN, PACKED, bias=True, dtype=dtype).cuda()
    with torch.no_grad():
        linear.weight.copy_(torch.randn_like(linear.weight))
        linear.bias.copy_(torch.randn_like(linear.bias))
    return linear


@requires_cuda
def test_packed_qkv_proj_bf16_matches_merged_linear_plus_cat():
    """The op output must be bit-identical to the fallback path it replaces
    (per-stream merged projection + one seq-dim cat): same GEMM problems on
    the same inputs, only the output placement differs."""
    txt, img, w_txt, b_txt, w_img, b_img = _make_inputs()
    out = torch.ops.trtllm_vgoa.packed_qkv_proj_bf16(txt, img, w_txt, b_txt, w_img, b_img)
    ref = torch.cat([F.linear(txt, w_txt, b_txt), F.linear(img, w_img, b_img)], dim=1)
    assert out.shape == (1, S_TXT + S_IMG, PACKED)
    assert out.dtype == txt.dtype
    assert torch.equal(out, ref)


@requires_cuda
def test_packed_qkv_proj_bf16_opcheck():
    """torch.library.opcheck validates the schema, fake impl, and
    functional (alias-free) contract the compiled path relies on."""
    args = _make_inputs()
    torch.library.opcheck(torch.ops.trtllm_vgoa.packed_qkv_proj_bf16, args)


@requires_cuda
def test_linear_supports_packed_addmm_census():
    eligible = Linear(HIDDEN, PACKED, bias=True, dtype=torch.bfloat16).cuda()
    assert linear_supports_packed_addmm(eligible, out_features=PACKED, in_features=HIDDEN)
    # Shape mismatch must be rejected.
    assert not linear_supports_packed_addmm(
        eligible, out_features=PACKED + HIDDEN, in_features=HIDDEN
    )
    # The op signature requires a bias tensor.
    no_bias = Linear(HIDDEN, PACKED, bias=False, dtype=torch.bfloat16).cuda()
    assert not linear_supports_packed_addmm(no_bias, out_features=PACKED, in_features=HIDDEN)
    # Only bf16 weights take the raw-addmm lane.
    fp16 = Linear(HIDDEN, PACKED, bias=True, dtype=torch.float16).cuda()
    assert not linear_supports_packed_addmm(fp16, out_features=PACKED, in_features=HIDDEN)


def test_linear_supports_packed_addmm_rejects_cpu_weights():
    cpu_linear = Linear(HIDDEN, PACKED, bias=True, dtype=torch.bfloat16)
    assert not linear_supports_packed_addmm(cpu_linear, out_features=PACKED, in_features=HIDDEN)


@requires_cuda
def test_recipe_dispatch_select_and_build():
    """The single dispatch seam: the census selects the bf16 recipe for an
    eligible merged-Linear pair, the builder reproduces the fallback path
    bit-identically from the Linears' native parameters, and any ineligible
    Linear in the pair drops dispatch to None (caller keeps the
    merged-forward + cat fallback)."""
    txt_proj = _make_merged_linear()
    img_proj = _make_merged_linear()
    recipe = select_packed_qkv_recipe((txt_proj, img_proj), out_features=PACKED, in_features=HIDDEN)
    assert recipe == "bf16"

    torch.manual_seed(1)
    txt = torch.randn(1, S_TXT, HIDDEN, device="cuda", dtype=torch.bfloat16)
    img = torch.randn(1, S_IMG, HIDDEN, device="cuda", dtype=torch.bfloat16)
    out = build_packed_qkv(recipe, txt, img, txt_proj, img_proj)
    ref = torch.cat(
        [
            F.linear(txt, txt_proj.weight, txt_proj.bias),
            F.linear(img, img_proj.weight, img_proj.bias),
        ],
        dim=1,
    )
    assert torch.equal(out, ref)

    fp16 = Linear(HIDDEN, PACKED, bias=True, dtype=torch.float16).cuda()
    assert (
        select_packed_qkv_recipe((txt_proj, fp16), out_features=PACKED, in_features=HIDDEN) is None
    )


@requires_cuda
def test_qwen_attention_compiled_path_engages_packed_recipe_and_matches_fallback() -> None:
    """The production model census and compiled dispatch must engage the leaf op.

    The QK-norm/RoPE epilogue is intentionally a no-op in this test so the
    assertion isolates the QKV projection selected inside the production
    ``QwenJointAttention._prepare_qkv_fused`` implementation.
    """
    torch.manual_seed(2)
    attention = (
        _QwenJointAttentionWithoutRope(
            dim=HIDDEN,
            num_attention_heads=2,
            attention_head_dim=HIDDEN // 2,
            dtype=torch.bfloat16,
            config=DiffusionModelConfig(),
        )
        .cuda()
        .eval()
    )
    attention.finalize_fused_qkv_pack()
    assert attention._qkv_pack_recipe == "bf16"

    hidden_states = torch.randn(1, S_IMG, HIDDEN, device="cuda", dtype=torch.bfloat16)
    encoder_hidden_states = torch.randn(1, S_TXT, HIDDEN, device="cuda", dtype=torch.bfloat16)
    unused_rope = (
        torch.empty(S_TXT + S_IMG, HIDDEN // 2, device="cuda", dtype=torch.bfloat16),
        torch.empty(S_TXT + S_IMG, HIDDEN // 2, device="cuda", dtype=torch.bfloat16),
    )

    selected_recipe = attention._qkv_pack_recipe
    attention._qkv_pack_recipe = None
    with torch.inference_mode():
        fallback = attention._prepare_qkv_fused(
            hidden_states,
            encoder_hidden_states,
            unused_rope,
            unused_rope,
        )

    attention._qkv_pack_recipe = selected_recipe
    compiled_prepare = torch.compile(attention._prepare_qkv_fused, fullgraph=True)
    with torch.inference_mode():
        # The first invocation performs the Inductor compile. ``fullgraph``
        # makes a graph break a hard failure rather than an eager fallback.
        compiled_prepare(hidden_states, encoder_hidden_states, unused_rope, unused_rope)
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profiler:
            packed = compiled_prepare(
                hidden_states,
                encoder_hidden_states,
                unused_rope,
                unused_rope,
            )
            torch.cuda.synchronize()

    event_names = {event.key for event in profiler.key_averages()}
    assert any("trtllm_vgoa::packed_qkv_proj_bf16" in name for name in event_names), (
        "compiled QwenJointAttention silently missed the selected packed-QKV recipe"
    )
    for packed_tensor, fallback_tensor in zip(packed, fallback):
        assert torch.equal(packed_tensor, fallback_tensor)
