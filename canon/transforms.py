from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch

from .model_utils import get_decoder_layers


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

    def to_dict(self) -> dict[str, float | int | str | bool]:
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
    cfg = model.config
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
    cfg = model.config
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
    return {
        "count": len(records),
        "max_cond_left": max(r.cond_left for r in records),
        "max_cond_right": max(r.cond_right for r in records),
        "max_cond_transform": max(r.cond_transform for r in records),
        "min_sv": min(r.sv_min for r in records),
        "max_sv": max(r.sv_max for r in records),
        "mean_balance_rel_before": math.fsum(r.balance_rel_before for r in records) / len(records),
        "mean_balance_rel_after": math.fsum(r.balance_rel_after for r in records) / len(records),
        "num_clipped": sum(1 for r in records if r.clipped),
    }
