import torch
import torch.nn as nn
from itertools import product
import torch.nn.functional as F

def pointwise_mse(logits, labels):
    scores = torch.sigmoid(logits)
    scores = scores.to(labels.dtype)
    return nn.MSELoss(reduction="mean")(scores, labels)


def pointwise_bce(logits, labels):
    return nn.BCEWithLogitsLoss(reduction="mean")(logits, labels)


def pairwise_ranknet(logits, labels, group_size):
    grouped_logits = logits.view(-1, group_size)
    grouped_labels = labels.view(-1, group_size)

    # here we generate every pair of indices from the range of document length in the batch
    document_pairs_candidates = list(product(range(grouped_labels.shape[1]), repeat=2))

    pairs_true = grouped_labels[:, document_pairs_candidates]
    selected_pred = grouped_logits[:, document_pairs_candidates]

    # here we calculate the relative true relevance of every candidate pair
    true_diffs = pairs_true[:, :, 0] - pairs_true[:, :, 1]
    pred_diffs = selected_pred[:, :, 0] - selected_pred[:, :, 1]

    # here we filter just the pairs that are 'positive' and did not involve a padded instance
    # we can do that since in the candidate pairs we had symetric pairs so we can stick with
    # positive ones for a simpler loss function formulation
    the_mask = (true_diffs > 0) & (~torch.isinf(true_diffs))

    pred_diffs = pred_diffs[the_mask]

    weight = None
    # true_pow_diffs = torch.pow(pairs_true[:, :, 0], 2) - torch.pow(pairs_true[:, :, 1], 2)
    abs_diff = torch.abs(true_diffs)
    weight = abs_diff[the_mask]

    # here we 'binarize' true relevancy diffs since for a pairwise loss we just need to know
    # whether one document is better than the other and not about the actual difference in
    # their relevancy levels
    true_diffs = (true_diffs > 0).type(torch.float32)
    true_diffs = true_diffs[the_mask]

    return nn.BCEWithLogitsLoss(reduction="mean", weight=weight)(pred_diffs, true_diffs)


def listwise_ce(logits, labels, group_size):
    grouped_logits = logits.view(-1, group_size)
    grouped_labels = labels.view(-1, group_size)

    # 只保留 label != 0 的部分进行 softmax, 这里默认 label 为 0 是负样本
    masked_labels = torch.where(grouped_labels != 0, grouped_labels, torch.tensor(float('-inf'), device=grouped_labels.device))
    grouped_labels = torch.softmax(masked_labels.detach(), dim=-1)
    
    loss = - torch.mean(
        torch.sum(
            grouped_labels * torch.log_softmax(grouped_logits, dim=-1), dim=-1
        )
    )
    return loss

def centered_mse(logits, labels, group_size):
    """
    Centered / relative regression loss.
    Preserves linear structure within each group.
    """
    grouped_logits = logits.view(-1, group_size)
    grouped_labels = labels.view(-1, group_size)

    # center within each group
    s_centered = grouped_logits - grouped_logits.mean(dim=1, keepdim=True)
    r_centered = grouped_labels - grouped_labels.mean(dim=1, keepdim=True)

    loss = ((s_centered - r_centered) ** 2).mean()
    return loss

def ranknet_with_centered_reg(
    logits,
    labels,
    group_size,
    lambda_centered=0.2,
):
    """
    Combined loss:
      RankNet (pairwise ordering)
    + Centered regression (relative distances)
    """
    loss_rank = pairwise_ranknet(logits, labels, group_size)
    loss_center = centered_mse(logits, labels, group_size)

    return (1 - lambda_centered) * loss_rank + lambda_centered * loss_center


# -----------------------------
# Attention scaling (adaptive)
# -----------------------------
def _maybe_scale_attention(
    psi: torch.Tensor,
    s: torch.Tensor,
    eps: float = 1e-8,
    # "not strict" thresholds:
    max_abs_threshold: float = 200.0,   # only scale if attention gets really big
    ratio_threshold: float = 20.0,      # only scale if attention dominates logits heavily
) -> torch.Tensor:
    """
    Adaptive scaling:
      - If psi is already small/reasonable -> return as-is.
      - If psi is huge (absolute or relative to logits) -> scale it down.

    Scale-down method:
      1) log1p compression
      2) per-group mean normalization (keeps magnitude ~ 1)
    """
    # psi: [B, G], s: [B, G]
    psi_abs_max = psi.abs().max()

    # Compare magnitude vs logits (robust-ish)
    s_scale = s.abs().mean(dim=1, keepdim=True).clamp_min(eps)         # [B, 1]
    psi_scale = psi.abs().mean(dim=1, keepdim=True)                    # [B, 1]
    ratio = (psi_scale / s_scale).median()                             # scalar

    needs_scale = (psi_abs_max > max_abs_threshold) or (ratio > ratio_threshold)

    if not needs_scale:
        return psi  # ✅ keep original

    # ✅ scale down (safe for nonnegative attention; still works if tiny negatives exist)
    psi2 = torch.clamp(psi, min=0.0)
    psi2 = torch.log1p(psi2)  # compress large values: 2000 -> ~7.6

    # normalize per group so typical magnitude is ~1
    denom = psi2.mean(dim=1, keepdim=True).clamp_min(eps)
    psi2 = psi2 / denom

    return psi2


def _group_ranks_from_labels(y: torch.Tensor) -> torch.Tensor:
    B, G = y.shape
    order = torch.argsort(y, dim=1, descending=True)
    r = torch.empty((B, G), device=y.device, dtype=torch.float32)
    ar = torch.arange(1, G + 1, device=y.device, dtype=torch.float32).unsqueeze(0).expand(B, G)
    r.scatter_(1, order, ar)
    return r


def pairwise_ranknet_yap_neg(
    logits: torch.Tensor,
    labels: torch.Tensor,
    yaps: torch.Tensor,
    train_group_size: int,
    lambda_yap: float = 0.3,
    reduction: str = "mean",
) -> torch.Tensor:
    G = int(train_group_size)
    assert logits.numel() % G == 0, f"logits length {logits.numel()} not divisible by group_size {G}"
    B = logits.numel() // G

    s = logits.view(B, G).float()
    y = labels.view(B, G).float()
    psi = yaps.view(B, G).float()

    # ✅ adaptive scaling (only if needed)
    psi = _maybe_scale_attention(psi, s)

    r = _group_ranks_from_labels(y)

    s_i = s.unsqueeze(2)       # [B, G, 1]
    s_j = s.unsqueeze(1)       # [B, 1, G]
    psi_j = psi.unsqueeze(1)   # [B, 1, G]

    y_i = y.unsqueeze(2)
    y_j = y.unsqueeze(1)
    mask = (y_i > y_j)

    r_i = r.unsqueeze(2)
    r_j = r.unsqueeze(1)
    w = (r_j - r_i).abs()

    delta = (s_j + lambda_yap * psi_j) - s_i
    per_pair = w * F.softplus(delta)
    per_pair = per_pair * mask.to(per_pair.dtype)

    if reduction == "sum":
        return per_pair.sum()
    elif reduction == "mean":
        denom = mask.sum().clamp_min(1).to(per_pair.dtype)
        return per_pair.sum() / denom
    else:
        return per_pair


def pairwise_ranknet_yap_pos(
    logits: torch.Tensor,
    labels: torch.Tensor,
    yaps: torch.Tensor,
    train_group_size: int,
    lambda_yap: float = 0.3,
    reduction: str = "mean",
) -> torch.Tensor:
    G = int(train_group_size)
    assert logits.numel() % G == 0, f"logits length {logits.numel()} not divisible by group_size {G}"
    B = logits.numel() // G

    s = logits.view(B, G).float()
    y = labels.view(B, G).float()
    psi = yaps.view(B, G).float()

    # ✅ adaptive scaling (only if needed)
    psi = _maybe_scale_attention(psi, s)

    r = _group_ranks_from_labels(y)

    s_i = s.unsqueeze(2)       # [B, G, 1]
    s_j = s.unsqueeze(1)       # [B, 1, G]
    psi_i = psi.unsqueeze(2)   # [B, G, 1]

    y_i = y.unsqueeze(2)
    y_j = y.unsqueeze(1)
    mask = (y_i > y_j)

    r_i = r.unsqueeze(2)
    r_j = r.unsqueeze(1)
    w = (r_j - r_i).abs()

    # POSITIVE side only
    delta = s_j - (s_i - lambda_yap * psi_i)

    per_pair = w * F.softplus(delta)
    per_pair = per_pair * mask.to(per_pair.dtype)

    if reduction == "sum":
        return per_pair.sum()
    else:  # mean default
        denom = mask.sum().clamp_min(1).to(per_pair.dtype)
        return per_pair.sum() / denom


def pairwise_ranknet_yap_both(
    logits: torch.Tensor,
    labels: torch.Tensor,
    yaps: torch.Tensor,
    train_group_size: int,
    lambda_yap: float = 0.3,
    reduction: str = "mean",
) -> torch.Tensor:
    G = int(train_group_size)
    assert logits.numel() % G == 0, f"logits length {logits.numel()} not divisible by group_size {G}"
    B = logits.numel() // G

    s = logits.view(B, G).float()
    y = labels.view(B, G).float()
    psi = yaps.view(B, G).float()

    # ✅ adaptive scaling (only if needed)
    psi = _maybe_scale_attention(psi, s)

    r = _group_ranks_from_labels(y)

    s_i = s.unsqueeze(2)
    s_j = s.unsqueeze(1)
    psi_i = psi.unsqueeze(2)
    psi_j = psi.unsqueeze(1)

    y_i = y.unsqueeze(2)
    y_j = y.unsqueeze(1)
    mask = (y_i > y_j)

    r_i = r.unsqueeze(2)
    r_j = r.unsqueeze(1)
    w = (r_j - r_i).abs()

    delta = (s_j + lambda_yap * psi_j) - (s_i - lambda_yap * psi_i)

    per_pair = w * F.softplus(delta)
    per_pair = per_pair * mask.to(per_pair.dtype)

    if reduction == "sum":
        return per_pair.sum()
    elif reduction == "mean":
        denom = mask.sum().clamp_min(1).to(per_pair.dtype)
        return per_pair.sum() / denom
    else:
        return per_pair



def pairwise_soft_ranknet(
    logits: torch.Tensor,
    labels: torch.Tensor,
    group_size: int,
    tau: float = 1.0,          # label-temperature (controls how "hard" targets are)
    eps: float = 1e-8,
):
    """
    Single integrated loss (no lambda, no extra term):
      - prediction: q_ij = sigmoid(s_i - s_j)
      - target:     p_ij = sigmoid((r_i - r_j) / tau)

    This encodes BOTH:
      - ranking (sign of r_i - r_j)
      - relevance magnitude (size of r_i - r_j -> how close p_ij is to 0/1)
    """

    s = logits.view(-1, group_size)             # [B, G]
    r = labels.view(-1, group_size)             # [B, G]

    valid = torch.isfinite(r)                   # [B, G]
    valid_pair = valid.unsqueeze(2) & valid.unsqueeze(1)  # [B, G, G]

    # pairwise diffs
    s_diff = s.unsqueeze(2) - s.unsqueeze(1)    # [B, G, G]
    r_diff = r.unsqueeze(2) - r.unsqueeze(1)    # [B, G, G]

    # drop diagonal (i == j) because diff is 0 and adds useless terms
    G = group_size
    diag = torch.eye(G, device=s.device, dtype=torch.bool).unsqueeze(0)  # [1, G, G]
    mask = valid_pair & (~diag)

    if not mask.any():
        return s.sum() * 0.0

    # soft targets from label gaps
    # clamp tau for safety
    tau = max(float(tau), eps)
    p = torch.sigmoid(r_diff / tau)             # [B, G, G] in (0,1)
    p = p[mask]

    # predicted pairwise probability
    q_logit = s_diff[mask]                      # logits for sigmoid
    # binary CE with soft targets is valid: BCE(q_logit, p)
    loss = F.binary_cross_entropy_with_logits(q_logit, p, reduction="mean")
    return loss

if __name__ == "__main__":
    torch.manual_seed(42)  # 固定随机种子以获得可复现结果
    logits = torch.randn(12, requires_grad=True)  # 生成 3*4 个随机 logit
    labels = torch.tensor([1, 0, 2, 0, 3, 0, 1, 0, 2, 0, 3, 0], dtype=torch.float)  # 定义标签，其中 0 表示负样本
    group_size = 4  # 每组 4 个样本
    
    loss = listwise_ce(logits, labels, group_size)
    print("Loss:", loss.item())
