"""Cupra adapter for DeepGEMM's SM90 packed-NVFP4 grouped GEMM."""

import deep_gemm
import torch


def prepare(w_packed, block_scale, global_scale, m, n, k, e):
    if m <= 128:
        k_blocks = k // 128
        canonical_vectors = w_packed.view(e, n, k_blocks, 4, 16)
        codes = torch.stack(
            (canonical_vectors & 0x0F, canonical_vectors >> 4), dim=-1
        ).flatten(-2)
        code_octets = codes.view(e, n, k_blocks, 4, 4, 8)
        first = code_octets[..., :4].to(torch.int64)
        second = code_octets[..., 4:].to(torch.int64)
        packed_words = torch.zeros_like(first[..., 0])
        for i in range(4):
            packed_words |= (first[..., i] & 0x7) << (4 * i)
            packed_words |= (first[..., i] & 0x8) << (4 + 8 * i)
            packed_words |= (second[..., i] & 0x7) << (16 + 4 * i)
            packed_words |= (second[..., i] & 0x8) << (8 * i)
        bit_woven_vectors = (
            packed_words.to(torch.int32).contiguous().view(w_packed.dtype)
        )
        packed_tiles = (
            bit_woven_vectors
            .permute(0, 2, 3, 1, 4)
            .contiguous()
        )
        scale_tiles = (
            block_scale.view(e, n, k_blocks, 2, 4)
            .permute(0, 2, 3, 1, 4)
            .contiguous()
            .view(w_packed.dtype)
        )
        n_tiles = n // 64
        packed_tiles = (
            packed_tiles
            .view(e, k_blocks, 4, n_tiles, 64, 16)
            .permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .flatten(3)
        )
        scale_tiles = (
            scale_tiles
            .view(e, k_blocks, 2, n_tiles, 64, 4)
            .permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .flatten(3)
        )
        # Exact-byte N64 tile layout. Each 32-bit word bit-weaves eight FP4 codes:
        # K[0:4] magnitudes and K[4:8] magnitudes are already two PRMT
        # selectors, while their sign bits already occupy (or shift directly
        # into) FP8 byte-sign positions. No DP4A selector packing is needed.
        # Within each contiguous 4,608-byte tile, four coalesced
        # [N64, 16-byte] vectors precede two [N64, 4-byte] scale vectors.
        fused = torch.cat(
            (packed_tiles, scale_tiles), dim=3
        ).view(e, k_blocks, 72, n)
        return [fused, block_scale.new_empty((0,)), global_scale]

    scale_k = k // 16
    # Model-load-time layout: for a fixed K group, adjacent N rows are adjacent
    # in memory. Producer warps can therefore fetch packed weights and scales
    # with coalesced transactions while the timed kernel still performs the
    # packed-FP4 -> transient WGMMA-input expansion itself on chip.
    w_prepacked = (
        w_packed.view(e, n, scale_k, 8)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    scale_prepacked = block_scale.permute(0, 2, 1).contiguous()
    return [w_prepacked, scale_prepacked, global_scale]


def run(c, a, a_scale, prepared, offsets, m, n, k, e):
    del m, n, k, e
    w_packed, block_scale, global_scale = prepared
    deep_gemm.m_grouped_nvfp4_gemm_nt_contiguous(
        a, a_scale, w_packed, block_scale, global_scale, c, offsets
    )
