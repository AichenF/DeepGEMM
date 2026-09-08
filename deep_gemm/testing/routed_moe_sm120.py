"""Deterministic DeepSeek-V4-Flash W4A8 fixtures for SM120 tests.

The tensors returned here use the public ``fp8_fp4_routed_moe_sm120`` ABI.
Raw launcher tests may adapt these tensors to their internal ABI, but benchmark
drivers should consume them without reaching into the extension module.
"""

from dataclasses import dataclass

import torch

DeviceLike = int | str | torch.device

RECIPE_ID = "dsv4-w4a8-distinct-k32-v1"
WORLD_SIZE = 8
EXPERTS = 256
LOCAL_EXPERTS = EXPERTS // WORLD_SIZE
TOP_K = 6
HIDDEN = 4096
INTERMEDIATE = 2048
MAX_ROWS = 8192
TASK_ROWS = 128
MMA_TILE_N = 128
K_GROUP = 32
ACTIVATION_CLAMP = 10.0
FAST_MATH = True
FP8_CODES = (0x30, 0x38, 0x3C, 0x40, 0xB0, 0xB8)
ROUTE_WEIGHTS = (0.3125, 0.25, 0.1875, 0.125, 0.078125, 0.046875)
_SIGNED_FP4_CODES = (1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15)
_POSITIVE_FP4_CODES = (1, 2, 3, 4, 5, 6, 7)
_SIGNED_FP4_VALUES = (
    0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)
_POSITIVE_FP4_VALUES = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

__all__ = [
    "ACTIVATION_CLAMP",
    "CONTRACT",
    "EXPERTS",
    "FAST_MATH",
    "FP8_CODES",
    "HIDDEN",
    "INTERMEDIATE",
    "K_GROUP",
    "LOCAL_EXPERTS",
    "MAX_ROWS",
    "MMA_TILE_N",
    "RECIPE_ID",
    "ROUTE_WEIGHTS",
    "TASK_ROWS",
    "TOP_K",
    "WORLD_SIZE",
    "DSV4W4A8Inputs",
    "DSV4W4A8Weights",
    "ModelContract",
    "deterministic_fp4_value",
    "deterministic_input_encoding",
    "deterministic_route_expert",
    "deterministic_weight_exponents",
    "epoch_slots",
    "make_dsv4_w4a8_inputs",
    "make_dsv4_w4a8_weights",
]


@dataclass(frozen=True)
class ModelContract:
    ep: int = WORLD_SIZE
    hidden: int = HIDDEN
    intermediate: int = INTERMEDIATE
    experts: int = EXPERTS
    top_k: int = TOP_K
    w1: str = "gate+up"
    w2: str = "down"
    activation: str = "MXFP8 E4M3, K32"
    weight: str = "MXFP4 E2M1, K32"
    boundary: str = (
        "W1 BF16 -> FP32 SwiGLU -> FP32 route weight -> K32 requant -> W2 BF16"
    )


CONTRACT = ModelContract()


@dataclass(frozen=True)
class DSV4W4A8Inputs:
    x: torch.Tensor
    x_scales: torch.Tensor
    topk_indices: torch.Tensor
    topk_weights: torch.Tensor


@dataclass(frozen=True)
class DSV4W4A8Weights:
    w1_weight: torch.Tensor
    w1_scales: torch.Tensor
    w2_weight: torch.Tensor
    w2_scales: torch.Tensor

    @property
    def w1_up_gate(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.w1_weight, self.w1_scales

    @property
    def w2_down(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.w2_weight, self.w2_scales


def epoch_slots(epochs: list[int] | tuple[int, ...]) -> list[int]:
    return [epoch & 1 for epoch in epochs]


def deterministic_route_expert(
    source_rank: int,
    token: int,
    route_slot: int,
) -> int:
    return (token * 17 + route_slot * 53 + source_rank * 97) % EXPERTS


def deterministic_input_encoding(
    source_rank: int,
    token: int,
    device: DeviceLike,
) -> tuple[torch.Tensor, torch.Tensor]:
    groups = torch.arange(HIDDEN // K_GROUP, device=device)
    code_table = torch.tensor(FP8_CODES, dtype=torch.uint8, device=device)
    codes = code_table[(groups + source_rank + token) % len(FP8_CODES)]
    exponents = (119 + (groups + 2 * source_rank + token) % 3).to(torch.uint8)
    return codes[:, None].expand(-1, K_GROUP).contiguous(), exponents


def deterministic_weight_exponents(
    global_expert: int,
    projection: str,
    device: DeviceLike,
) -> torch.Tensor:
    if projection == "w1":
        groups, offset = HIDDEN // K_GROUP, 0
    elif projection == "w2":
        groups, offset = INTERMEDIATE // K_GROUP, 1
    else:
        raise ValueError(f"unknown projection {projection!r}")
    group = torch.arange(groups, device=device)
    return (119 + (group + global_expert + offset) % 3).to(torch.uint8)


def deterministic_fp4_value(
    global_expert: int,
    projection: str,
    logical_index: torch.Tensor,
) -> torch.Tensor:
    if projection == "gate":
        pattern = (logical_index + global_expert) % len(_SIGNED_FP4_CODES)
        table = _SIGNED_FP4_VALUES
    elif projection == "up":
        pattern = (3 * logical_index + global_expert) % len(_SIGNED_FP4_CODES)
        table = _SIGNED_FP4_VALUES
    elif projection == "down":
        pattern = (logical_index + global_expert) % len(_POSITIVE_FP4_CODES)
        table = _POSITIVE_FP4_VALUES
    else:
        raise ValueError(f"unknown projection {projection!r}")
    return torch.tensor(table, dtype=torch.float32, device=logical_index.device)[pattern]


def _validate_rank(rank: int) -> None:
    if not isinstance(rank, int) or not 0 <= rank < WORLD_SIZE:
        raise ValueError(f"rank must be in [0, {WORLD_SIZE}), got {rank!r}")


def _pack_scale_words(exponents: torch.Tensor) -> torch.Tensor:
    if exponents.shape[-1] % 4:
        raise ValueError("UE8M0 exponent count must be divisible by four")
    return exponents.contiguous().view(torch.int32)


def make_dsv4_w4a8_inputs(
    rank: int,
    active_rows: int,
    device: DeviceLike,
) -> DSV4W4A8Inputs:
    """Create deterministic public-ABI activations and balanced top-k routes."""

    _validate_rank(rank)
    if not isinstance(active_rows, int) or not 1 <= active_rows <= MAX_ROWS:
        raise ValueError(f"active_rows must be in [1, {MAX_ROWS}], got {active_rows!r}")

    tokens = torch.arange(active_rows, device=device)[:, None]
    slots = torch.arange(TOP_K, device=device)[None, :]
    topk_indices = (
        (tokens * 17 + slots * 53 + rank * 97) % EXPERTS
    ).to(torch.int64)
    route_weights = torch.tensor(ROUTE_WEIGHTS, dtype=torch.float32, device=device)

    groups = torch.arange(HIDDEN // K_GROUP, device=device)[None, :]
    code_table = torch.tensor(FP8_CODES, dtype=torch.uint8, device=device)
    codes = code_table[(groups + rank + tokens) % len(FP8_CODES)]
    x = codes[:, :, None].expand(-1, -1, K_GROUP).reshape(active_rows, HIDDEN)
    scale_exp = (119 + (groups + 2 * rank + tokens) % 3).to(torch.uint8)
    x_scales = _pack_scale_words(scale_exp).reshape(active_rows, HIDDEN // 128)
    return DSV4W4A8Inputs(
        x=x.contiguous(),
        x_scales=x_scales.contiguous(),
        topk_indices=topk_indices.contiguous(),
        topk_weights=route_weights.expand(active_rows, -1).contiguous(),
    )


def make_dsv4_w4a8_weights(
    rank: int,
    device: DeviceLike,
) -> DSV4W4A8Weights:
    """Create deterministic packed MXFP4 weights and UE8M0 K32 scales."""

    _validate_rank(rank)
    global_experts = rank * LOCAL_EXPERTS + torch.arange(
        LOCAL_EXPERTS,
        device=device,
    )

    w1 = torch.empty(
        (LOCAL_EXPERTS, 2 * INTERMEDIATE, HIDDEN // 2),
        dtype=torch.uint8,
        device=device,
    )
    logical = torch.arange(INTERMEDIATE, device=device)
    physical_up = (logical // 8) * 16 + logical % 8 + 8
    physical_gate = physical_up - 8
    signed_codes = torch.tensor(_SIGNED_FP4_CODES, dtype=torch.uint8, device=device)
    gate_nibble = signed_codes[
        (logical[None, :] + global_experts[:, None]) % len(_SIGNED_FP4_CODES)
    ]
    up_nibble = signed_codes[
        (3 * logical[None, :] + global_experts[:, None])
        % len(_SIGNED_FP4_CODES)
    ]
    w1[:, physical_gate, :] = (gate_nibble | (gate_nibble << 4))[:, :, None]
    w1[:, physical_up, :] = (up_nibble | (up_nibble << 4))[:, :, None]

    positive_codes = torch.tensor(_POSITIVE_FP4_CODES, dtype=torch.uint8, device=device)
    packed_index = torch.arange(INTERMEDIATE // 2, device=device)
    low = positive_codes[
        (2 * packed_index[None, :] + global_experts[:, None])
        % len(_POSITIVE_FP4_CODES)
    ]
    high = positive_codes[
        (2 * packed_index[None, :] + 1 + global_experts[:, None])
        % len(_POSITIVE_FP4_CODES)
    ]
    packed = low | (high << 4)
    output = torch.arange(HIDDEN, device=device)
    sign = (((output[None, :] // 64 + global_experts[:, None]) & 1) * 0x88).to(torch.uint8)
    w2 = (packed[:, None, :] ^ sign[:, :, None]).contiguous()

    def scales(projection: str, width: int, k: int) -> torch.Tensor:
        groups = k // K_GROUP
        k_blocks = k // 128
        offset = 0 if projection == "w1" else 1
        exponent = (
            119
            + (
                torch.arange(groups, device=device)[None, :]
                + global_experts[:, None]
                + offset
            )
            % 3
        ).to(torch.uint8)
        packed = exponent.reshape(LOCAL_EXPERTS, k_blocks, 4)
        packed = packed[:, :, None, :].expand(-1, -1, width, -1).contiguous()
        return packed.view(torch.int32).reshape(LOCAL_EXPERTS, k_blocks, width)

    return DSV4W4A8Weights(
        w1_weight=w1.view(torch.int8),
        w1_scales=scales("w1", 2 * INTERMEDIATE, HIDDEN),
        w2_weight=w2.view(torch.int8),
        w2_scales=scales("w2", HIDDEN, INTERMEDIATE),
    )
