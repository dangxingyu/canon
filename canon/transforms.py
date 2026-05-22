from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch

from .model_utils import get_decoder_layers, get_text_config


@dataclass
class BalanceRecord:
    kind: str
    layer: int
    group: int
    cond_left: float
    cond_right: float
    cond_transform: float
    sv_min: float
    sv_max: float
    balance_rel_before: float
    balance_rel_after: float
    clipped: bool
    # Optional fields populated when applicable (orthogonal canon defines all four;
    # SPD canon currently leaves them at None to keep the existing schema stable).
    commutator_rel_before: float | None = None
    commutator_rel_after: float | None = None
    subspace_overlap_before: float | None = None
    subspace_overlap_after: float | None = None
    ortho_residual: float | None = None

    def to_dict(self) -> dict[str, float | int | str | bool | None]:
        return asdict(self)


def _sym(x: torch.Tensor) -> torch.Tensor:
    return (x + x.T) * 0.5


def _spd_eigh(x: torch.Tensor, floor: float = 1e-18) -> tuple[torch.Tensor, torch.Tensor]:
    evals, evecs = torch.linalg.eigh(_sym(x))
    return torch.clamp(evals, min=floor), evecs


def _spd_power(x: torch.Tensor, power: float, floor: float = 1e-18) -> torch.Tensor:
    evals, evecs = _spd_eigh(x, floor=floor)
    return (evecs * evals.pow(power).unsqueeze(0)) @ evecs.T


def _cond_from_evals(evals: torch.Tensor) -> float:
    vals = torch.clamp(evals.detach().double(), min=1e-30)
    return float((vals.max() / vals.min()).item())


def _relative_gap(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = torch.linalg.norm(a, ord="fro") + torch.linalg.norm(b, ord="fro") + 1e-30
    return float((torch.linalg.norm(a - b, ord="fro") / denom).item())


def _commutator_rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """||A B - B A||_F / (||A||_F * ||B||_F + eps). Measures co-diagonalizability."""
    denom = float((torch.linalg.norm(a, ord="fro") * torch.linalg.norm(b, ord="fro")).item()) + 1e-30
    return float(torch.linalg.norm(a @ b - b @ a, ord="fro").item()) / denom


def _subspace_overlap(a: torch.Tensor, b: torch.Tensor, k: int | None = None) -> float:
    """Mean squared cosine of principal angles between the top-k eigensubspaces of A and B.

    1.0 = subspaces identical; 0.0 = orthogonal. k defaults to head_dim // 2.
    """
    d = a.shape[0]
    if k is None:
        k = max(1, d // 2)
    _, U_a = torch.linalg.eigh(_sym(a))
    _, U_b = torch.linalg.eigh(_sym(b))
    # eigh returns ascending; we want top-k (largest eigenvalues, last k columns)
    Ua_top = U_a[:, -k:]
    Ub_top = U_b[:, -k:]
    M = Ua_top.T @ Ub_top  # (k, k)
    return float((torch.linalg.norm(M, ord="fro") ** 2 / k).item())


def solve_orthogonal_balancer(
    left: torch.Tensor,
    right: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float | bool]]:
    """Find orthogonal P that aligns eigenbasis of ``left`` onto eigenbasis of ``right``.

    Both inputs are SPD head_dim x head_dim Gram matrices. The function-preserving
    gauge ``V <- P V, O <- O P^T`` is restricted to orthogonal P, which always
    preserves head-dim RMS norms (so it survives v_norm with `with_scale=False`).

    The chosen ``P`` is Procrustes-style: ``P = U_B S U_A^T`` where U_A, U_B are
    the descending-eigenvalue eigenbases of left/right and S is a sign diagonal
    chosen to make ``det(P) = +1`` so that consecutive applications compose
    without reflection.
    """
    A = _sym(left.double())
    B = _sym(right.double())
    evals_a, U_a = _spd_eigh(A)
    evals_b, U_b = _spd_eigh(B)
    # _spd_eigh returns ascending — flip to descending
    U_a = U_a.flip(-1)
    U_b = U_b.flip(-1)
    evals_a = evals_a.flip(-1)
    evals_b = evals_b.flip(-1)

    # Raw Procrustes: P = U_b U_a^T maps A's eigenbasis to B's eigenbasis with matched ordering.
    P = U_b @ U_a.T
    # Pin det(P) = +1 (proper rotation, no reflection) by flipping one column sign of U_b
    # if needed. Reflection is also orthogonal and function-preserving, but using a proper
    # rotation makes records like cond_transform reliably equal to 1.
    if float(torch.linalg.det(P).item()) < 0:
        U_b_fixed = U_b.clone()
        U_b_fixed[:, -1] = -U_b_fixed[:, -1]
        P = U_b_fixed @ U_a.T

    # Sanity: how close is P to truly orthogonal (should be machine eps in fp64).
    d = P.shape[0]
    ortho_residual = float((P @ P.T - torch.eye(d, dtype=P.dtype)).norm().item())

    A_new = P @ A @ P.T
    return P, {
        "cond_left": _cond_from_evals(evals_a),
        "cond_right": _cond_from_evals(evals_b),
        "cond_transform": 1.0,  # orthogonal P has unit singular values
        "sv_min": 1.0,
        "sv_max": 1.0,
        "clipped": False,
        "ortho_residual": ortho_residual,
        "frob_gap_before": _relative_gap(A, B),
        "frob_gap_after": _relative_gap(A_new, B),  # invariant under conjugation by same P
        "subspace_overlap_before": _subspace_overlap(A, B),
        "subspace_overlap_after": _subspace_overlap(A_new, B),
        "commutator_rel_before": _commutator_rel(A, B),
        "commutator_rel_after": _commutator_rel(A_new, B),
    }


def solve_spd_balancer(
    left: torch.Tensor,
    right: torch.Tensor,
    min_sv: float,
    max_sv: float,
) -> tuple[torch.Tensor, dict[str, float | bool]]:
    """Solve P left P^T ~= P^{-T} right P^{-1} for SPD P.

    With X = P^T P, the exact SPD solution satisfies X left X = right.
    We return a spectrum-clipped SPD P to control numerical drift.
    """

    left = _sym(left.double())
    right = _sym(right.double())
    left_evals, _ = _spd_eigh(left)
    right_evals, _ = _spd_eigh(right)
    left_half = _spd_power(left, 0.5)
    left_inv_half = _spd_power(left, -0.5)
    middle = _sym(left_half @ right @ left_half)
    middle_half = _spd_power(middle, 0.5)
    x = _sym(left_inv_half @ middle_half @ left_inv_half)
    p = _spd_power(x, 0.5)

    p_evals, p_evecs = _spd_eigh(p)
    clipped_evals = torch.clamp(p_evals, min=min_sv, max=max_sv)
    clipped = bool(torch.any(clipped_evals != p_evals).item())
    p = (p_evecs * clipped_evals.unsqueeze(0)) @ p_evecs.T

    return p, {
        "cond_left": _cond_from_evals(left_evals),
        "cond_right": _cond_from_evals(right_evals),
        "cond_transform": _cond_from_evals(clipped_evals),
        "sv_min": float(clipped_evals.min().item()),
        "sv_max": float(clipped_evals.max().item()),
        "clipped": clipped,
    }


@torch.no_grad()
def apply_gqa_value_output_covariance_balance(
    model,
    cov_left: list[torch.Tensor],
    cov_right: list[torch.Tensor],
    *,
    lambda_scale: float = 1e-6,
    min_sv: float = 0.25,
    max_sv: float = 4.0,
    kind: str = "gqa_vo_actgrad",
    trace_normalize: bool = False,
) -> list[BalanceRecord]:
    cfg = get_text_config(model)
    num_heads = int(cfg.num_attention_heads)
    num_kv_heads = int(cfg.num_key_value_heads)
    head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // num_heads))
    heads_per_kv = num_heads // num_kv_heads
    records: list[BalanceRecord] = []

    for layer_idx, layer in enumerate(get_decoder_layers(model)):
        attn = layer.self_attn
        v_weight = attn.v_proj.weight
        o_weight = attn.o_proj.weight
        device = v_weight.device
        out_dtype = v_weight.dtype

        for kv_idx in range(num_kv_heads):
            C0 = cov_left[layer_idx][kv_idx].detach().double().cpu()
            F0 = cov_right[layer_idx][kv_idx].detach().double().cpu()
            if trace_normalize:
                F0 = F0 * (torch.trace(C0) / torch.clamp(torch.trace(F0), min=1e-30))
            scale = float(((torch.trace(C0) + torch.trace(F0)) / (2 * head_dim)).item())
            lam = max(1e-18, lambda_scale * max(scale, 1e-18))
            eye = torch.eye(head_dim, dtype=torch.float64)
            C = C0 + lam * eye
            F = F0 + lam * eye
            before = _relative_gap(C, F)
            P, info = solve_spd_balancer(C, F, min_sv=min_sv, max_sv=max_sv)
            Pinv = torch.linalg.inv(P)
            after = _relative_gap(P @ C @ P.T, Pinv.T @ F @ Pinv)

            v_rows = slice(kv_idx * head_dim, (kv_idx + 1) * head_dim)
            P_dev = P.to(device=device, dtype=out_dtype)
            Pinv_dev = Pinv.to(device=o_weight.device, dtype=o_weight.dtype)
            v_weight[v_rows, :] = P_dev @ v_weight[v_rows, :]
            for local_head in range(heads_per_kv):
                head_idx = kv_idx * heads_per_kv + local_head
                o_cols = slice(head_idx * head_dim, (head_idx + 1) * head_dim)
                o_weight[:, o_cols] = o_weight[:, o_cols] @ Pinv_dev

            records.append(
                BalanceRecord(
                    kind=kind,
                    layer=layer_idx,
                    group=kv_idx,
                    cond_left=float(info["cond_left"]),
                    cond_right=float(info["cond_right"]),
                    cond_transform=float(info["cond_transform"]),
                    sv_min=float(info["sv_min"]),
                    sv_max=float(info["sv_max"]),
                    balance_rel_before=before,
                    balance_rel_after=after,
                    clipped=bool(info["clipped"]),
                )
            )
    return records


@torch.no_grad()
def apply_gqa_value_output_norm_balance(
    model,
    *,
    lambda_scale: float = 1e-6,
    min_sv: float = 0.25,
    max_sv: float = 4.0,
) -> list[BalanceRecord]:
    cfg = get_text_config(model)
    num_heads = int(cfg.num_attention_heads)
    num_kv_heads = int(cfg.num_key_value_heads)
    head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // num_heads))
    heads_per_kv = num_heads // num_kv_heads
    if num_heads % num_kv_heads != 0:
        raise ValueError(f"num_attention_heads={num_heads} is not divisible by num_key_value_heads={num_kv_heads}")

    records: list[BalanceRecord] = []
    for layer_idx, layer in enumerate(get_decoder_layers(model)):
        attn = layer.self_attn
        v_weight = attn.v_proj.weight
        o_weight = attn.o_proj.weight
        device = v_weight.device
        out_dtype = v_weight.dtype

        for kv_idx in range(num_kv_heads):
            v_rows = slice(kv_idx * head_dim, (kv_idx + 1) * head_dim)
            V = v_weight[v_rows, :].detach().double().cpu()
            o_blocks = []
            for local_head in range(heads_per_kv):
                head_idx = kv_idx * heads_per_kv + local_head
                o_cols = slice(head_idx * head_dim, (head_idx + 1) * head_dim)
                o_blocks.append(o_weight[:, o_cols].detach().double().cpu())

            A0 = V @ V.T
            B0 = sum(O.T @ O for O in o_blocks)
            scale = float(((torch.trace(A0) + torch.trace(B0)) / (2 * head_dim)).item())
            lam = max(1e-12, lambda_scale * max(scale, 1e-12))
            eye = torch.eye(head_dim, dtype=torch.float64)
            A = A0 + lam * eye
            B = B0 + lam * eye
            before = _relative_gap(A, B)

            P, info = solve_spd_balancer(A, B, min_sv=min_sv, max_sv=max_sv)
            Pinv = torch.linalg.inv(P)
            after = _relative_gap(P @ A @ P.T, Pinv.T @ B @ Pinv)

            P_dev = P.to(device=device, dtype=out_dtype)
            Pinv_dev = Pinv.to(device=o_weight.device, dtype=o_weight.dtype)
            v_weight[v_rows, :] = P_dev @ v_weight[v_rows, :]
            for local_head in range(heads_per_kv):
                head_idx = kv_idx * heads_per_kv + local_head
                o_cols = slice(head_idx * head_dim, (head_idx + 1) * head_dim)
                o_weight[:, o_cols] = o_weight[:, o_cols] @ Pinv_dev

            records.append(
                BalanceRecord(
                    kind="gqa_vo_norm",
                    layer=layer_idx,
                    group=kv_idx,
                    cond_left=float(info["cond_left"]),
                    cond_right=float(info["cond_right"]),
                    cond_transform=float(info["cond_transform"]),
                    sv_min=float(info["sv_min"]),
                    sv_max=float(info["sv_max"]),
                    balance_rel_before=before,
                    balance_rel_after=after,
                    clipped=bool(info["clipped"]),
                )
            )

    return records


@torch.no_grad()
def apply_gqa_value_output_orthogonal_balance(
    model,
    *,
    lambda_scale: float = 1e-6,
    skip_kv_shared: bool = True,
) -> list[BalanceRecord]:
    """Orthogonal V/O canon: V <- P V, O <- O P^T with P proper-rotation.

    Function-preserving even when v_norm is applied along head_dim with no scale
    (Gemma-3n / Gemma-4 case). The chosen P is Procrustes-style, aligning the
    eigenbasis of V V^T to that of sum_h O_h^T O_h. Spectra are unitarily
    invariant under this gauge, so the Frobenius gap ||A-B||_F is unchanged --
    the metrics that move are commutator_rel and subspace_overlap.

    ``skip_kv_shared`` skips layers whose attention reuses K/V from an earlier
    layer (Gemma-3n attribute ``self_attn.is_kv_shared_layer``); their v_proj
    weights are dead at inference and applying canon on them would conflict with
    the canon already applied to the source layer's v_proj. Layers without this
    attribute are always processed.
    """
    cfg = get_text_config(model)
    num_heads = int(cfg.num_attention_heads)
    num_kv_heads = int(cfg.num_key_value_heads)
    head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // num_heads))
    heads_per_kv = num_heads // num_kv_heads
    if num_heads % num_kv_heads != 0:
        raise ValueError(f"num_attention_heads={num_heads} is not divisible by num_key_value_heads={num_kv_heads}")

    # If any layer in this model shares K/V cross-layer, dispatch to the Gemma
    # variant which gauges *all* sink layers' o_proj alongside the source's.
    layers_list = list(get_decoder_layers(model))
    if any(bool(getattr(layer.self_attn, "is_kv_shared_layer", False)) for layer in layers_list):
        return apply_gqa_value_output_orthogonal_balance_gemma(
            model, lambda_scale=lambda_scale
        )

    records: list[BalanceRecord] = []
    for layer_idx, layer in enumerate(layers_list):
        attn = layer.self_attn
        if skip_kv_shared and bool(getattr(attn, "is_kv_shared_layer", False)):
            continue
        if not hasattr(attn, "v_proj"):
            # Defensive: Gemma-3n / Gemma-4 drop the v_proj module on KV-shared
            # layers entirely, so skip even if is_kv_shared_layer is not set.
            continue
        v_weight = attn.v_proj.weight
        o_weight = attn.o_proj.weight
        device = v_weight.device
        out_dtype = v_weight.dtype
        # Per-layer head_dim (Gemma-3n / Gemma-4 vary head_dim between sliding /
        # full attention layers). Derive it from v_proj's first dim so we don't
        # have to consult ``layer_types`` here.
        layer_head_dim = v_weight.shape[0] // num_kv_heads
        layer_q_head_dim = o_weight.shape[1] // num_heads
        if layer_head_dim != layer_q_head_dim:
            raise ValueError(
                f"layer {layer_idx}: v_proj head_dim {layer_head_dim} != o_proj head_dim {layer_q_head_dim}"
            )

        for kv_idx in range(num_kv_heads):
            v_rows = slice(kv_idx * layer_head_dim, (kv_idx + 1) * layer_head_dim)
            V = v_weight[v_rows, :].detach().double().cpu()
            o_blocks = []
            for local_head in range(heads_per_kv):
                head_idx = kv_idx * heads_per_kv + local_head
                o_cols = slice(head_idx * layer_head_dim, (head_idx + 1) * layer_head_dim)
                o_blocks.append(o_weight[:, o_cols].detach().double().cpu())

            A0 = V @ V.T
            B0 = sum(O.T @ O for O in o_blocks)
            scale = float(((torch.trace(A0) + torch.trace(B0)) / (2 * layer_head_dim)).item())
            lam = max(1e-12, lambda_scale * max(scale, 1e-12))
            eye = torch.eye(layer_head_dim, dtype=torch.float64)
            A = A0 + lam * eye
            B = B0 + lam * eye

            P, info = solve_orthogonal_balancer(A, B)
            # For orthogonal P, P^{-1} = P^T (det = +1 by construction in the solver).
            Pinv = P.T

            P_dev = P.to(device=device, dtype=out_dtype)
            Pinv_dev = Pinv.to(device=o_weight.device, dtype=o_weight.dtype)
            v_weight[v_rows, :] = P_dev @ v_weight[v_rows, :]
            for local_head in range(heads_per_kv):
                head_idx = kv_idx * heads_per_kv + local_head
                o_cols = slice(head_idx * layer_head_dim, (head_idx + 1) * layer_head_dim)
                o_weight[:, o_cols] = o_weight[:, o_cols] @ Pinv_dev

            records.append(
                BalanceRecord(
                    kind="gqa_vo_orth",
                    layer=layer_idx,
                    group=kv_idx,
                    cond_left=float(info["cond_left"]),
                    cond_right=float(info["cond_right"]),
                    cond_transform=float(info["cond_transform"]),
                    sv_min=float(info["sv_min"]),
                    sv_max=float(info["sv_max"]),
                    balance_rel_before=float(info["frob_gap_before"]),
                    balance_rel_after=float(info["frob_gap_after"]),
                    clipped=bool(info["clipped"]),
                    commutator_rel_before=float(info["commutator_rel_before"]),
                    commutator_rel_after=float(info["commutator_rel_after"]),
                    subspace_overlap_before=float(info["subspace_overlap_before"]),
                    subspace_overlap_after=float(info["subspace_overlap_after"]),
                    ortho_residual=float(info["ortho_residual"]),
                )
            )

    return records


def _gemma_kv_sink_map(layers, num_hidden_layers: int, layer_types=None) -> dict[int, list[int]]:
    """For Gemma-3n / Gemma-4: return source_layer_idx -> [sink_layer_idx, ...].

    Gemma 3n exposes ``self_attn.kv_shared_layer_index`` as the source layer for
    each sink. Gemma 4 instead stashes K/V in ``shared_kv_states[layer_type]`` at
    forward time and does NOT expose the source as an attribute — we have to
    infer it from ``cfg.layer_types``: the source for each sink is the last
    non-shared layer of the same layer_type before the KV-share boundary.

    Also supports the Gemma-3n attribute path for backward compatibility.
    """
    sinks: dict[int, list[int]] = {i: [] for i in range(num_hidden_layers)}

    # Try the Gemma-3n attribute path first.
    found_via_attr = False
    for layer_idx, layer in enumerate(layers):
        attn = layer.self_attn
        if bool(getattr(attn, "is_kv_shared_layer", False)):
            src = getattr(attn, "kv_shared_layer_index", None)
            if isinstance(src, int) and src >= 0:
                sinks[src].append(layer_idx)
                found_via_attr = True
    if found_via_attr:
        return sinks

    # Gemma-4 fallback: use layer_types to find source-by-type. The source for
    # each sink layer is the LAST non-shared layer of matching layer_type.
    if layer_types is None:
        return sinks  # cannot derive without layer_types
    first_shared = next(
        (i for i, layer in enumerate(layers) if bool(getattr(layer.self_attn, "is_kv_shared_layer", False))),
        len(layers),
    )
    prev_types = layer_types[:first_shared]
    # For each layer_type that appears in prev_types, find its last occurrence.
    last_of_type: dict[str, int] = {}
    for i, t in enumerate(prev_types):
        last_of_type[t] = i
    for layer_idx, layer in enumerate(layers):
        attn = layer.self_attn
        if not bool(getattr(attn, "is_kv_shared_layer", False)):
            continue
        lt = layer_types[layer_idx] if layer_idx < len(layer_types) else None
        # Some Gemma 4 modules attach `layer_type` directly; prefer it if present.
        lt = getattr(attn, "layer_type", lt)
        if lt in last_of_type:
            sinks[last_of_type[lt]].append(layer_idx)
    return sinks


@torch.no_grad()
def apply_gqa_value_output_orthogonal_balance_gemma(
    model,
    *,
    lambda_scale: float = 1e-6,
) -> list[BalanceRecord]:
    """Gemma-3n / Gemma-4 aware orthogonal V/O canon.

    Differences vs ``apply_gqa_value_output_orthogonal_balance``:
      - Loops only over **source** layers (those whose self_attn computes K/V).
      - Builds ``B = Σ_h O_h^T O_h`` over the source's own Q heads PLUS every
        Q head of every KV-shared layer reading from this source.
      - Applies ``O ← O P^T`` to all of those o_proj weights too, so the
        attention output remains function-preserving end-to-end despite the
        cross-layer KV sharing.
    Function-preserving on Gemma-4 E2B's hybrid sliding/full attention (v_norm
    without scale tolerates orthogonal P).
    """
    cfg = get_text_config(model)
    num_heads = int(cfg.num_attention_heads)
    num_kv_heads = int(cfg.num_key_value_heads)
    head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // num_heads))
    heads_per_kv = num_heads // num_kv_heads
    if num_heads % num_kv_heads != 0:
        raise ValueError(f"num_attention_heads={num_heads} is not divisible by num_key_value_heads={num_kv_heads}")

    layers = list(get_decoder_layers(model))
    layer_types = getattr(cfg, "layer_types", None)
    sinks = _gemma_kv_sink_map(layers, len(layers), layer_types=layer_types)

    records: list[BalanceRecord] = []
    for layer_idx, layer in enumerate(layers):
        attn = layer.self_attn
        if bool(getattr(attn, "is_kv_shared_layer", False)):
            continue
        if not hasattr(attn, "v_proj"):
            continue
        v_weight = attn.v_proj.weight
        source_o_weight = attn.o_proj.weight
        device = v_weight.device
        out_dtype = v_weight.dtype
        layer_head_dim = v_weight.shape[0] // num_kv_heads
        # Verify the source's o_proj head_dim matches.
        if source_o_weight.shape[1] // num_heads != layer_head_dim:
            raise ValueError(
                f"layer {layer_idx}: v_proj head_dim {layer_head_dim} != "
                f"o_proj head_dim {source_o_weight.shape[1] // num_heads}"
            )
        # Collect all o_proj weights that consume this source layer's V.
        sink_layer_indices = sinks.get(layer_idx, [])
        sink_o_weights = [layers[i].self_attn.o_proj.weight for i in sink_layer_indices]
        for i, ow in zip(sink_layer_indices, sink_o_weights):
            sink_head_dim = ow.shape[1] // num_heads
            if sink_head_dim != layer_head_dim:
                raise ValueError(
                    f"sink layer {i} reading from source {layer_idx} has head_dim "
                    f"{sink_head_dim}, expected {layer_head_dim}"
                )

        for kv_idx in range(num_kv_heads):
            v_rows = slice(kv_idx * layer_head_dim, (kv_idx + 1) * layer_head_dim)
            V = v_weight[v_rows, :].detach().double().cpu()

            def _gather_o_blocks(ow):
                blocks = []
                for local_head in range(heads_per_kv):
                    head_idx = kv_idx * heads_per_kv + local_head
                    o_cols = slice(head_idx * layer_head_dim, (head_idx + 1) * layer_head_dim)
                    blocks.append(ow[:, o_cols].detach().double().cpu())
                return blocks

            source_blocks = _gather_o_blocks(source_o_weight)
            sink_blocks_per_layer = [_gather_o_blocks(ow) for ow in sink_o_weights]

            A0 = V @ V.T
            B0 = sum(O.T @ O for O in source_blocks)
            for sink_blocks in sink_blocks_per_layer:
                B0 = B0 + sum(O.T @ O for O in sink_blocks)
            scale = float(((torch.trace(A0) + torch.trace(B0)) / (2 * layer_head_dim)).item())
            lam = max(1e-12, lambda_scale * max(scale, 1e-12))
            eye = torch.eye(layer_head_dim, dtype=torch.float64)
            A = A0 + lam * eye
            B = B0 + lam * eye

            P, info = solve_orthogonal_balancer(A, B)
            Pinv = P.T

            P_dev = P.to(device=device, dtype=out_dtype)
            v_weight[v_rows, :] = P_dev @ v_weight[v_rows, :]

            # Apply Pinv to every o_proj that reads from this V: source + all sinks.
            for ow in (source_o_weight, *sink_o_weights):
                Pinv_dev = Pinv.to(device=ow.device, dtype=ow.dtype)
                for local_head in range(heads_per_kv):
                    head_idx = kv_idx * heads_per_kv + local_head
                    o_cols = slice(head_idx * layer_head_dim, (head_idx + 1) * layer_head_dim)
                    ow[:, o_cols] = ow[:, o_cols] @ Pinv_dev

            records.append(
                BalanceRecord(
                    kind="gqa_vo_orth_gemma",
                    layer=layer_idx,
                    group=kv_idx,
                    cond_left=float(info["cond_left"]),
                    cond_right=float(info["cond_right"]),
                    cond_transform=float(info["cond_transform"]),
                    sv_min=float(info["sv_min"]),
                    sv_max=float(info["sv_max"]),
                    balance_rel_before=float(info["frob_gap_before"]),
                    balance_rel_after=float(info["frob_gap_after"]),
                    clipped=bool(info["clipped"]),
                    commutator_rel_before=float(info["commutator_rel_before"]),
                    commutator_rel_after=float(info["commutator_rel_after"]),
                    subspace_overlap_before=float(info["subspace_overlap_before"]),
                    subspace_overlap_after=float(info["subspace_overlap_after"]),
                    ortho_residual=float(info["ortho_residual"]),
                )
            )

    return records


@torch.no_grad()
def apply_swiglu_mlp_activation_gradient_balance(
    model,
    hidden_second: list[torch.Tensor],
    grad_second: list[torch.Tensor],
    *,
    min_scale: float = 0.25,
    max_scale: float = 4.0,
    eps: float = 1e-18,
    normalize_geomean: bool = False,
) -> list[BalanceRecord]:
    records: list[BalanceRecord] = []
    for layer_idx, layer in enumerate(get_decoder_layers(model)):
        mlp = layer.mlp
        up = mlp.up_proj.weight
        down = mlp.down_proj.weight
        h2 = hidden_second[layer_idx].detach().float().to(up.device)
        g2 = grad_second[layer_idx].detach().float().to(up.device)
        raw = ((g2 + eps) / (h2 + eps)).pow(0.25)
        if normalize_geomean:
            raw = raw / torch.exp(torch.mean(torch.log(torch.clamp(raw, min=eps))))
        scale = torch.clamp(raw, min=min_scale, max=max_scale).to(device=up.device, dtype=up.dtype)
        clipped = bool(torch.any((raw < min_scale) | (raw > max_scale)).item())

        up.mul_(scale[:, None])
        down.div_(scale[None, :].to(device=down.device, dtype=down.dtype))

        records.append(
            BalanceRecord(
                kind="swiglu_mlp_actgrad_gmean" if normalize_geomean else "swiglu_mlp_actgrad",
                layer=layer_idx,
                group=-1,
                cond_left=float((torch.clamp(h2, min=eps).max() / torch.clamp(h2, min=eps).min()).item()),
                cond_right=float((torch.clamp(g2, min=eps).max() / torch.clamp(g2, min=eps).min()).item()),
                cond_transform=float((scale.float().max() / torch.clamp(scale.float().min(), min=eps)).item()),
                sv_min=float(scale.float().min().item()),
                sv_max=float(scale.float().max().item()),
                balance_rel_before=float(torch.std(torch.log(raw.float() + eps)).item()),
                balance_rel_after=float(torch.std(torch.log(scale.float() + eps)).item()),
                clipped=clipped,
            )
        )
    return records


@torch.no_grad()
def apply_swiglu_mlp_norm_balance(
    model,
    *,
    min_scale: float = 0.25,
    max_scale: float = 4.0,
    eps: float = 1e-12,
) -> list[BalanceRecord]:
    records: list[BalanceRecord] = []
    for layer_idx, layer in enumerate(get_decoder_layers(model)):
        mlp = layer.mlp
        up = mlp.up_proj.weight
        down = mlp.down_proj.weight
        up_norm = torch.linalg.norm(up.detach().float(), dim=1)
        down_norm = torch.linalg.norm(down.detach().float(), dim=0)
        raw = torch.sqrt((down_norm + eps) / (up_norm + eps))
        scale = torch.clamp(raw, min=min_scale, max=max_scale).to(device=up.device, dtype=up.dtype)
        clipped = bool(torch.any((raw < min_scale) | (raw > max_scale)).item())

        up.mul_(scale[:, None])
        down.div_(scale[None, :].to(device=down.device, dtype=down.dtype))

        records.append(
            BalanceRecord(
                kind="swiglu_mlp_norm",
                layer=layer_idx,
                group=-1,
                cond_left=float((up_norm.max() / torch.clamp(up_norm.min(), min=eps)).item()),
                cond_right=float((down_norm.max() / torch.clamp(down_norm.min(), min=eps)).item()),
                cond_transform=float((scale.float().max() / torch.clamp(scale.float().min(), min=eps)).item()),
                sv_min=float(scale.float().min().item()),
                sv_max=float(scale.float().max().item()),
                balance_rel_before=float((torch.std(torch.log(raw.float() + eps))).item()),
                balance_rel_after=float((torch.std(torch.log(scale.float() + eps))).item()),
                clipped=clipped,
            )
        )
    return records


def summarize_records(records: list[BalanceRecord]) -> dict[str, float | int]:
    if not records:
        return {"count": 0}

    def _opt_mean(name: str) -> float | None:
        vals = [getattr(r, name) for r in records if getattr(r, name) is not None]
        if not vals:
            return None
        return math.fsum(vals) / len(vals)

    summary: dict[str, float | int | None] = {
        "count": len(records),
        "max_cond_left": max(r.cond_left for r in records),
        "max_cond_right": max(r.cond_right for r in records),
        "max_cond_transform": max(r.cond_transform for r in records),
        "min_sv": min(r.sv_min for r in records),
        "max_sv": max(r.sv_max for r in records),
        "mean_balance_rel_before": math.fsum(r.balance_rel_before for r in records) / len(records),
        "mean_balance_rel_after": math.fsum(r.balance_rel_after for r in records) / len(records),
        "num_clipped": sum(1 for r in records if r.clipped),
        "mean_commutator_rel_before": _opt_mean("commutator_rel_before"),
        "mean_commutator_rel_after": _opt_mean("commutator_rel_after"),
        "mean_subspace_overlap_before": _opt_mean("subspace_overlap_before"),
        "mean_subspace_overlap_after": _opt_mean("subspace_overlap_after"),
        "max_ortho_residual": max(
            (r.ortho_residual for r in records if r.ortho_residual is not None),
            default=None,
        ),
    }
    return summary
