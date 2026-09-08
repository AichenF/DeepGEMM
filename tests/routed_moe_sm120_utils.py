"""Deterministic DeepSeek-V4-Flash W4A8 fixtures for tests and benchmarks.

The tensors returned here use the public ``fp8_fp4_routed_moe_sm120`` ABI.
This is intentionally not a pytest module. It defines the structured public-API
fixture shared by the routed-MoE test and benchmark.
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
    "expected_local_work",
    "expected_route",
    "expected_tokens",
    "make_dsv4_w4a8_inputs",
    "make_dsv4_w4a8_weights",
    "source_order_combine",
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


def source_order_combine(partials: torch.Tensor) -> torch.Tensor:
    """Accumulate original route slots in FP32, then round once to BF16."""

    if partials.ndim != 3 or partials.shape[1] != TOP_K:
        raise ValueError(f"partials must have shape [tokens, {TOP_K}, hidden]")
    combined = torch.zeros(
        (partials.shape[0], partials.shape[2]),
        dtype=torch.float32,
        device=partials.device,
    )
    for route_slot in range(TOP_K):
        combined += partials[:, route_slot].float()
    return combined.to(torch.bfloat16)


def _physical_gate_up(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    if gate.shape != up.shape or gate.shape[-1] != INTERMEDIATE:
        raise ValueError("W1 must contain one gate and one up projection")
    output = torch.empty(
        (*gate.shape[:-1], 2 * INTERMEDIATE),
        dtype=gate.dtype,
        device=gate.device,
    )
    logical = torch.arange(INTERMEDIATE, device=gate.device)
    physical_gate = (logical // 8) * 16 + logical % 8
    output[..., physical_gate] = gate
    output[..., physical_gate + 8] = up
    return output


def _split_physical_gate_up(w1_bf16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    logical = torch.arange(INTERMEDIATE, device=w1_bf16.device)
    physical_gate = (logical // 8) * 16 + logical % 8
    return w1_bf16[..., physical_gate], w1_bf16[..., physical_gate + 8]


def _requantize_k32(weighted: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grouped = weighted.reshape(weighted.shape[0], INTERMEDIATE // K_GROUP, K_GROUP)
    raw_scale = (grouped.abs().amax(dim=2) * (1.0 / 448.0)).contiguous()
    bits = raw_scale.view(torch.int32)
    exponent = (
        ((bits >> 23) & 255) + (((bits & 0x7FFFFF) + 0x7FFFFF) >> 23)
    ).clamp(max=254).to(torch.uint8)
    inverse = ((254 - exponent.to(torch.int32)) << 23).view(torch.float32)
    fp8 = (grouped * inverse[:, :, None]).to(torch.float8_e4m3fn).view(torch.uint8)
    scale = torch.ldexp(
        torch.ones_like(exponent, dtype=torch.float32),
        exponent.to(torch.int32) - 127,
    )
    dequantized = fp8.view(torch.float8_e4m3fn).float() * scale[:, :, None]
    return fp8.reshape(weighted.shape), exponent, dequantized.reshape(weighted.shape)


def expected_route(
    source_rank: int,
    token: int,
    route_slot: int,
    global_expert: int,
    device: DeviceLike,
    *,
    round_swiglu_output: bool = False,
) -> dict[str, torch.Tensor]:
    """Compute one route from the fixture recipe, without kernel intermediates."""

    codes, input_exponents = deterministic_input_encoding(source_rank, token, device)
    input_scales = torch.ldexp(
        torch.ones_like(input_exponents, dtype=torch.float32),
        input_exponents.to(torch.int32) - 127,
    )
    x_group_sum = (codes.view(torch.float8_e4m3fn).float() * input_scales[:, None]).sum(dim=1)
    w1_exponents = deterministic_weight_exponents(global_expert, "w1", device)
    w1_scales = torch.ldexp(
        torch.ones_like(w1_exponents, dtype=torch.float32),
        w1_exponents.to(torch.int32) - 127,
    )
    common = (x_group_sum * w1_scales).sum()
    logical = torch.arange(INTERMEDIATE, device=device)
    gate = (common * deterministic_fp4_value(global_expert, "gate", logical)).to(torch.bfloat16)
    up = (common * deterministic_fp4_value(global_expert, "up", logical)).to(torch.bfloat16)
    w1_bf16 = _physical_gate_up(gate[None, :], up[None, :])

    gate_f32, up_f32 = _split_physical_gate_up(w1_bf16)
    gate_f32 = gate_f32.float().clamp(max=ACTIVATION_CLAMP)
    up_f32 = up_f32.float().clamp(min=-ACTIVATION_CLAMP, max=ACTIVATION_CLAMP)
    silu = gate_f32 * (1.0 / (1.0 + torch.exp2(-gate_f32 * 1.4426950408889634)))
    weighted = silu * up_f32 * float(ROUTE_WEIGHTS[route_slot])
    if round_swiglu_output:
        weighted = weighted.to(torch.bfloat16).float()
    intermediate_fp8, intermediate_scale, intermediate = _requantize_k32(weighted)

    w2_exponents = deterministic_weight_exponents(global_expert, "w2", device)
    w2_scales = torch.ldexp(
        torch.ones_like(w2_exponents, dtype=torch.float32),
        w2_exponents.to(torch.int32) - 127,
    )
    down_weight = deterministic_fp4_value(global_expert, "down", logical)
    down = (
        intermediate.reshape(INTERMEDIATE // K_GROUP, K_GROUP)
        * w2_scales[:, None]
        * down_weight.reshape(INTERMEDIATE // K_GROUP, K_GROUP)
    ).sum()
    output = torch.arange(HIDDEN, device=device)
    output_sign = torch.where(
        ((output // 64 + global_expert) & 1) == 0,
        1.0,
        -1.0,
    )
    return {
        "w1_bf16": w1_bf16[0],
        "intermediate_fp8": intermediate_fp8[0],
        "intermediate_scale": intermediate_scale[0],
        "w2_bf16": (down * output_sign).to(torch.bfloat16),
    }


def expected_tokens(
    rank: int,
    tokens: tuple[int, ...],
    device: DeviceLike,
) -> torch.Tensor:
    rows = []
    for token in tokens:
        routes = []
        for slot in range(TOP_K):
            expert = deterministic_route_expert(rank, token, slot)
            routes.append(expected_route(rank, token, slot, expert, device)["w2_bf16"])
        rows.append(torch.stack(routes))
    return source_order_combine(torch.stack(rows))


def expected_local_work(rank: int, active_rows: int) -> tuple[int, int]:
    local_counts = [0] * LOCAL_EXPERTS
    for source in range(WORLD_SIZE):
        for token in range(active_rows):
            for slot in range(TOP_K):
                expert = deterministic_route_expert(source, token, slot)
                if expert // LOCAL_EXPERTS == rank:
                    local_counts[expert % LOCAL_EXPERTS] += 1
    return sum(local_counts), sum(
        (count + TASK_ROWS - 1) // TASK_ROWS for count in local_counts
    )
