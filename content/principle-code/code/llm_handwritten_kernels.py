"""
LLM 算法工程师面试常见手撕代码。

这份文件保持“能手写、能解释、能运行”的粒度，覆盖：
- 解码采样：top-k / top-p / temperature
- 基础算子：stable softmax / cross entropy / LayerNorm / RMSNorm
- 后训练：SFT shift loss / DPO / PPO clip / GAE / GRPO
- 架构：scaled dot-product attention / MHA / RoPE
- MTP：multi-token shifted labels / 多头 future-token loss / speculative verify
- Agent：极简 ReAct 工具调用循环

运行检查：
    python content/principle-code/code/llm_handwritten_kernels.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    raise SystemExit(
        "This script requires PyTorch. Install torch in your interview-practice "
        "environment before running the quick checks."
    ) from exc


def top_k_filtering(
    logits: torch.Tensor,
    top_k: int,
    temperature: float = 1.0,
    filter_value: float = -float("inf"),
) -> torch.Tensor:
    """
    Top-k 采样过滤：每个位置只保留概率最高的 k 个 token。

    面试讲解：
    - temperature 先作用在 logits 上，temperature 越小分布越尖锐。
    - 被过滤 token 的 logit 置为 -inf，softmax 后概率变成 0。
    - top_k <= 0 时等价于不做 top-k，只做温度缩放。
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if top_k <= 0:
        return logits / temperature

    logits = logits / temperature
    top_k = min(top_k, logits.size(-1))

    # threshold 是第 k 大 logit；小于它的 token 全部被过滤掉。
    threshold = torch.topk(logits, k=top_k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < threshold, filter_value)


def top_p_filtering(
    logits: torch.Tensor,
    top_p: float,
    temperature: float = 1.0,
    min_tokens_to_keep: int = 1,
    filter_value: float = -float("inf"),
) -> torch.Tensor:
    """
    Top-p / nucleus 采样：保留累计概率刚超过 p 的最小 token 集合。

    面试讲解：
    - top-k 是固定保留 k 个，top-p 是按分布形状动态决定保留数量。
    - 分布很尖锐时保留 token 少，分布很平时保留 token 多。
    - 右移 remove mask 是为了保留“第一个让累计概率超过 p 的 token”。
    """
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    logits = logits / temperature
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = sorted_probs.cumsum(dim=-1)

    sorted_remove = cumulative_probs > top_p
    sorted_remove[..., :min_tokens_to_keep] = False

    # 保留第一个越过 top_p 的 token，否则概率质量可能不足 p。
    sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
    sorted_remove[..., 0] = False

    # sorted_remove 是排序后的位置，需要 scatter 回原始 vocab 顺序。
    remove = torch.zeros_like(sorted_remove).scatter(dim=-1, index=sorted_indices, src=sorted_remove)
    return logits.masked_fill(remove, filter_value)


def stable_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    数值稳定 softmax：先减去最大值，避免 exp(logit) 溢出。

    softmax(x) = exp(x_i) / sum(exp(x_j))。
    减去同一个常数不改变结果，但能把最大 logit 变成 0。
    """
    shifted = logits - logits.max(dim=dim, keepdim=True).values
    exp = shifted.exp()
    return exp / exp.sum(dim=dim, keepdim=True)


def cross_entropy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    手写交叉熵：先算 log_softmax，再取目标类别的负 log prob。

    输入：
    - logits: [batch, num_classes]
    - targets: [batch]，每个位置是类别下标。
    """
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    target_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return -target_log_probs.mean()


class ManualLayerNorm(nn.Module):
    """
    LayerNorm：对最后一维做均值方差归一化，再乘 gamma 加 beta。

    和 BatchNorm 的关键区别：LayerNorm 不依赖 batch 统计，更适合 Transformer 序列建模。
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta


class ManualRMSNorm(nn.Module):
    """
    RMSNorm：只用均方根归一化，不减均值。

    面试讲解：RMSNorm 比 LayerNorm 少了 mean-centering 和 beta，计算更轻，很多 LLM 使用它。
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return self.gamma * x / rms


def sft_cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Causal LM 的 SFT loss：用第 t 位 logits 预测第 t+1 位 label。

    关键点：
    - shift_logits = logits[:, :-1, :]。
    - shift_labels = labels[:, 1:]。
    - ignore_index 通常用于屏蔽 prompt token，只训练 assistant answer token。
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Scaled Dot-Product Attention。

    输入形状：[batch, heads, seq, head_dim]。
    除以 sqrt(head_dim) 是为了避免 qk 点积过大导致 softmax 饱和、梯度变小。
    """
    scale = q.size(-1) ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if mask is not None:
        # mask=0 的位置不可见，例如 causal mask 会屏蔽未来 token。
        scores = scores.masked_fill(mask == 0, -float("inf"))
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, v), attn


class MultiHeadSelfAttention(nn.Module):
    """
    最小多头自注意力实现。

    手撕重点：
    1. 一次线性层生成 Q/K/V，再 reshape 成多头。
    2. 每个 head 独立做 attention。
    3. 拼回 hidden_size 后过输出投影。
    """

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        batch, seq_len, hidden = x.shape
        qkv = self.qkv(x).view(batch, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        # 从 [B, S, H, D] 转成 [B, H, S, D]，方便每个 head 单独算 attention。
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]

        mask = None
        if causal:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
            mask = mask.view(1, 1, seq_len, seq_len)

        context, _ = scaled_dot_product_attention(q, k, v, mask)
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, hidden)
        return self.out(context)


def apply_rope(x: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """
    RoPE 旋转位置编码：对 hidden dim 的偶数/奇数维成对旋转。

    输入形状：[..., seq, dim]，dim 必须为偶数。
    直觉：不同位置使用不同旋转角，让 attention 点积天然包含相对位置信息。
    """
    seq_len, dim = x.shape[-2], x.shape[-1]
    if dim % 2 != 0:
        raise ValueError("RoPE requires an even hidden dimension")

    device = x.device
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    freqs = torch.einsum("s,d->sd", positions, inv_freq)
    cos = freqs.cos().view(*([1] * (x.ndim - 2)), seq_len, dim // 2)
    sin = freqs.sin().view(*([1] * (x.ndim - 2)), seq_len, dim // 2)

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos
    return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)



def vit_patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    ViT patchify 最小实现。
    images: [batch, channels, height, width]
    返回: [batch, num_patches, channels * patch_size * patch_size]

    面试抓手：ViT 不是直接把整张图送进 Transformer，而是先切成固定大小 patch，
    每个 patch 展平成一个 token，再接线性投影和位置编码。
    """
    if images.ndim != 4:
        raise ValueError("images must be [batch, channels, height, width]")
    batch, channels, height, width = images.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("height and width must be divisible by patch_size")

    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    # [B, C, H/P, W/P, P, P] -> [B, H/P, W/P, C, P, P]
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    return patches.view(batch, -1, channels * patch_size * patch_size)


def clip_contrastive_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    CLIP 双向图文对比损失。
    image_features / text_features: [batch, hidden]

    面试抓手：batch 内第 i 张图和第 i 条文本是正样本，其余组合是负样本。
    图到文和文到图各做一次交叉熵，保证两个检索方向都对齐。
    """
    if image_features.shape != text_features.shape:
        raise ValueError("image_features and text_features must have the same shape")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    logits = image_features @ text_features.T / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    image_to_text = F.cross_entropy(logits, labels)
    text_to_image = F.cross_entropy(logits.T, labels)
    return (image_to_text + text_to_image) / 2


def mae_random_mask(
    patch_embeddings: torch.Tensor,
    mask_ratio: float = 0.75,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    MAE 随机 mask patch 的核心步骤。
    patch_embeddings: [batch, num_patches, hidden]
    返回 visible_patches、mask、restore_indices。

    面试抓手：MAE 的 encoder 只看未被 mask 的 patch，decoder 再根据 restore_indices
    把 token 放回原顺序去重建被遮挡 patch，因此训练效率比把所有 patch 都送进 encoder 更高。
    """
    if not 0 <= mask_ratio < 1:
        raise ValueError("mask_ratio must be in [0, 1)")
    batch, num_patches, hidden = patch_embeddings.shape
    keep = max(1, int(num_patches * (1 - mask_ratio)))

    noise = torch.rand(batch, num_patches, device=patch_embeddings.device)
    shuffle_indices = noise.argsort(dim=1)
    restore_indices = shuffle_indices.argsort(dim=1)
    keep_indices = shuffle_indices[:, :keep]

    visible = patch_embeddings.gather(
        dim=1,
        index=keep_indices.unsqueeze(-1).expand(-1, -1, hidden),
    )
    mask = torch.ones(batch, num_patches, device=patch_embeddings.device)
    mask[:, :keep] = 0
    mask = mask.gather(dim=1, index=restore_indices)
    return visible, mask, restore_indices


def diffusion_q_sample(
    x0: torch.Tensor,
    t: torch.Tensor,
    sqrt_alphas_cumprod: torch.Tensor,
    sqrt_one_minus_alphas_cumprod: torch.Tensor,
    noise: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    DDPM 前向加噪 q(x_t | x_0) 的最小实现。
    x0: 原始图像或 latent，形状可以是 [B, C, H, W]。
    t: [B]，每个样本对应的扩散时间步。

    公式：x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * epsilon。
    面试抓手：训练时通常让模型预测 epsilon，目标是把不同噪声强度下的去噪问题变成监督学习。
    """
    if noise is None:
        noise = torch.randn_like(x0)
    shape = (x0.size(0),) + (1,) * (x0.ndim - 1)
    sqrt_alpha = sqrt_alphas_cumprod[t].view(shape)
    sqrt_one_minus = sqrt_one_minus_alphas_cumprod[t].view(shape)
    return sqrt_alpha * x0 + sqrt_one_minus * noise, noise
def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """
    DPO loss：让策略模型相对参考模型更偏好 chosen，而不是 rejected。

    输入通常是整条 response 的 log probability sum。
    核心比较量：(policy_chosen - policy_rejected) - (ref_chosen - ref_rejected)。
    """
    chosen_logratios = policy_chosen_logps - ref_chosen_logps
    rejected_logratios = policy_rejected_logps - ref_rejected_logps
    logits = beta * (chosen_logratios - rejected_logratios)
    return -F.logsigmoid(logits).mean()


def ppo_clipped_policy_loss(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """
    PPO clipped surrogate loss。

    ratio = pi_new(a|s) / pi_old(a|s)。
    clip 的作用是限制单步策略更新幅度，避免 advantage 很大时策略崩掉。
    """
    ratio = torch.exp(new_logprobs - old_logprobs)
    unclipped = ratio * advantages
    clipped = ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages
    return -torch.min(unclipped, clipped).mean()


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor | None = None,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GAE：Generalized Advantage Estimation。

    rewards: [T]
    values: [T + 1]，多一个 bootstrap value。
    dones: [T]，1 表示终止状态，后续 value 不再 bootstrap。

    直觉：GAE 是多步 TD error 的指数加权和，lam 控制 bias-variance tradeoff。
    """
    if dones is None:
        dones = torch.zeros_like(rewards)
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros((), device=rewards.device, dtype=rewards.dtype)
    for t in reversed(range(rewards.numel())):
        non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * non_terminal - values[t]
        gae = delta + gamma * lam * non_terminal * gae
        advantages[t] = gae
    returns = advantages + values[:-1]
    return advantages, returns


def grpo_group_advantages(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    GRPO 组内 advantage：同一个 prompt 采样多条 response，在组内做 reward 标准化。

    输入形状：[num_prompts, group_size]。
    好处：不需要额外训练 value model，而是用组内相对好坏构造 advantage。
    """
    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, unbiased=False, keepdim=True)
    return (rewards - mean) / (std + eps)



def rloo_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """
    RLOO / REINFORCE Leave-One-Out advantage。
    rewards: [num_prompts, group_size]，同一行表示同一个 prompt 的多条 response reward。

    手写直觉：
    - 对第 i 条 response，不用全组均值做 baseline，而是只用“其他 response”的均值。
    - 这样可以避免样本自己的 reward 泄漏进 baseline，常用于 critic-free 的方差降低。
    - group_size 必须大于 1，否则没有可比较的其他样本。
    """
    if rewards.ndim != 2:
        raise ValueError("rewards must be [num_prompts, group_size]")
    group_size = rewards.size(-1)
    if group_size <= 1:
        raise ValueError("RLOO requires group_size > 1")

    leave_one_out_baseline = (rewards.sum(dim=-1, keepdim=True) - rewards) / (group_size - 1)
    return rewards - leave_one_out_baseline


def sequence_level_importance_ratio(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    response_mask: torch.Tensor,
    normalize_by_length: bool = True,
) -> torch.Tensor:
    """
    GSPO 直觉版 sequence-level importance ratio。
    输入形状: [batch, response_len]，logprobs 是每个生成 token 的 log probability。

    为什么有用：
    - 标准 PPO/GRPO 常在 token 粒度使用 ratio = exp(new-old)。
    - 长 CoT 或 MoE 场景中，token 级 ratio 容易在局部剧烈抖动。
    - GSPO 类思路把 ratio 聚合到 sequence 粒度，让“优化单位”更贴近序列级 reward。

    normalize_by_length=True 时使用平均 log-ratio，避免长回答天然累积更大的 log-ratio。
    """
    if new_logprobs.shape != old_logprobs.shape or new_logprobs.shape != response_mask.shape:
        raise ValueError("new_logprobs, old_logprobs and response_mask must share the same shape")

    mask = response_mask.to(new_logprobs.dtype)
    log_ratio_sum = ((new_logprobs - old_logprobs) * mask).sum(dim=-1)
    if normalize_by_length:
        denom = mask.sum(dim=-1).clamp_min(1.0)
        log_ratio_sum = log_ratio_sum / denom
    return torch.exp(log_ratio_sum)
def grpo_loss(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    rewards: torch.Tensor,
    clip_eps: float = 0.2,
    beta: float = 0.04,
) -> torch.Tensor:
    """
    极简 GRPO-style loss。

    形状：[num_prompts, group_size]，logprobs 是序列级 log probability sum。
    损失 = PPO clip 策略损失 + beta * reference KL 约束。
    """
    advantages = grpo_group_advantages(rewards)
    policy_loss = ppo_clipped_policy_loss(
        new_logprobs.reshape(-1),
        old_logprobs.reshape(-1),
        advantages.reshape(-1),
        clip_eps,
    )

    # k3 KL 估计：exp(ref-new) - (ref-new) - 1，非负且数值比较稳定。
    log_ratio_ref = ref_logprobs - new_logprobs
    k3_kl = torch.exp(log_ratio_ref) - log_ratio_ref - 1
    return policy_loss + beta * k3_kl.mean()




def dapo_clipped_policy_loss(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    clip_low: float = 0.2,
    clip_high: float = 0.28,
    loss_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    DAPO 的非对称 clip policy loss。

    与标准 PPO/GRPO 对称 clip 不同：
    - 下界 1-clip_low 仍然限制负向大更新，保证稳定性。
    - 上界 1+clip_high 可以设得更大，给正优势 token 更多探索空间。

    输入通常是 token-level logprob，loss_mask=1 表示该 token 参与训练。
    """
    ratio = torch.exp(new_logprobs - old_logprobs)
    unclipped = ratio * advantages
    clipped = ratio.clamp(1 - clip_low, 1 + clip_high) * advantages
    loss = -torch.min(unclipped, clipped)
    if loss_mask is None:
        return loss.mean()
    return (loss * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)


def filter_zero_variance_reward_groups(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    DAPO dynamic sampling 的核心直觉：过滤组内 reward 没有差异的 prompt。

    rewards: [num_prompts, group_size]。
    返回 bool mask: True 表示该 prompt 的 group 有有效相对优势。

    如果同一个 prompt 的所有回答全对或全错，组内 std 约等于 0，
    GRPO/DAPO 的 group-relative advantage 没有训练信号。
    """
    return rewards.std(dim=-1, unbiased=False) > eps


def token_mean_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    """
    token-level loss aggregation：把所有有效 token 放在同一分母下平均。

    DAPO / VeRL 文档里强调 token-mean 可以缓解长序列训练中的长度偏差。
    对比 seq-mean-token-sum：长回答会因为 token 更多而贡献更大梯度。
    """
    return (loss_mat * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)


def soft_overlong_penalty(
    lengths: torch.Tensor,
    max_length: int,
    buffer_length: int,
    penalty_factor: float = 1.0,
) -> torch.Tensor:
    """
    DAPO soft overlong punishment / reward shaping。

    当回复长度超过 max_length - buffer_length 后，线性增加负惩罚；
    超过 max_length 时达到 -penalty_factor。
    """
    if buffer_length <= 0:
        raise ValueError("buffer_length must be positive")
    expected = max_length - buffer_length
    exceed = (lengths - expected).clamp_min(0).float()
    penalty = -(exceed / float(buffer_length)).clamp_max(1.0) * penalty_factor
    return penalty


def token_level_forward_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """
    OPD / GKD OPD 的 token-level forward KL: KL(teacher || student)。

    student_logits / teacher_logits: [batch, seq, vocab]。
    loss_mask: [batch, seq]，只在 response token 上为 1。

    关键点：teacher_logits 要 detach，teacher 只提供监督，不参与反向传播。
    """
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits.detach(), dim=-1)
    teacher_probs = teacher_log_probs.exp()
    kl_per_token = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
    return (kl_per_token * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)


def topk_forward_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    loss_mask: torch.Tensor,
    topk: int = 32,
) -> torch.Tensor:
    """
    VeRL GKD OPD 常用的 teacher-top-k forward KL 近似。

    真实 full-vocab KL 很贵；推理 server 通常只容易返回 sampled token 和
    teacher top-k token 的 logprobs，所以用 teacher top-k 近似 forward KL。
    """
    teacher_log_probs = F.log_softmax(teacher_logits.detach(), dim=-1)
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    topk_logp, topk_ids = torch.topk(teacher_log_probs, k=min(topk, teacher_logits.size(-1)), dim=-1)
    teacher_topk_probs = topk_logp.exp()
    student_topk_logp = student_log_probs.gather(dim=-1, index=topk_ids)
    kl_per_token = (teacher_topk_probs * (topk_logp - student_topk_logp)).sum(dim=-1)
    return (kl_per_token * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)


def sampled_reverse_kl_reward(
    teacher_token_logprob: torch.Tensor,
    student_token_logprob: torch.Tensor,
    clamp: float | None = None,
) -> torch.Tensor:
    """
    PG OPD / sampled-token OPD 的 teacher reward。

    reward = stop_gradient(log teacher_prob(sampled token) - log student_prob(sampled token))。
    它是 reverse KL 单样本估计的负号形式，用作 policy-gradient reward。
    """
    reward = teacher_token_logprob.detach() - student_token_logprob.detach()
    if clamp is not None:
        reward = reward.clamp(min=-clamp, max=clamp)
    return reward


def pg_opd_loss(
    new_token_logprobs: torch.Tensor,
    old_token_logprobs: torch.Tensor,
    teacher_token_logprobs: torch.Tensor,
    loss_mask: torch.Tensor,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """
    最小 PG OPD loss：用 teacher-student log-ratio 作为 token reward，
    再套 PPO/GRPO-style ratio clip。

    注意：teacher reward 必须 stop-gradient，否则会破坏 PG OPD 的梯度路径。
    """
    rewards = sampled_reverse_kl_reward(teacher_token_logprobs, old_token_logprobs)
    ratio = torch.exp(new_token_logprobs - old_token_logprobs)
    unclipped = ratio * rewards
    clipped = ratio.clamp(1 - clip_eps, 1 + clip_eps) * rewards
    loss = -torch.min(unclipped, clipped)
    return (loss * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)


def power_opd_reward(
    teacher_token_logprob: torch.Tensor,
    student_token_logprob: torch.Tensor,
    alpha: float = 0.2,
) -> torch.Tensor:
    """
    PowerOPD sampled-token reward：用有界 power transform 替代无界 log-ratio。

    alpha 越接近 0，越接近 log-ratio；alpha > 0 时 reward 有界，
    可以降低 teacher/student 概率极小导致的极端梯度。
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    teacher_prob = teacher_token_logprob.exp()
    student_prob = student_token_logprob.exp()
    teacher_score = (teacher_prob.pow(alpha) - 1.0) / alpha
    student_score = (student_prob.pow(alpha) - 1.0) / alpha
    return teacher_score - student_score

@dataclass
class MiniDataProto:
    """VeRL DataProto 的面试版极简容器。

    真实 VeRL 的 DataProto 会携带 tensor batch、non-tensor metadata，
    并支持 split / dispatch / collect。这里保留最关键的直觉：
    每个 RL step 的 prompt、response、logprob、reward、advantage、mask
    都在同一个数据容器里流动，controller 通过 union 逐步补齐字段。
    """

    batch: Dict[str, torch.Tensor]
    meta: Dict[str, str] | None = None

    def union(self, other: "MiniDataProto") -> "MiniDataProto":
        """合并 worker 返回的新字段，模拟 DataProto.union。"""
        merged_batch = dict(self.batch)
        merged_batch.update(other.batch)
        merged_meta = dict(self.meta or {})
        merged_meta.update(other.meta or {})
        return MiniDataProto(merged_batch, merged_meta)


def mock_verl_rl_dataflow(
    prompts: torch.Tensor,
    rollout_fn: Callable[[MiniDataProto], MiniDataProto],
    actor_logprob_fn: Callable[[MiniDataProto], MiniDataProto],
    ref_logprob_fn: Callable[[MiniDataProto], MiniDataProto],
    reward_fn: Callable[[MiniDataProto], MiniDataProto],
    advantage_fn: Callable[[MiniDataProto], MiniDataProto],
    update_actor_fn: Callable[[MiniDataProto], Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """手写 VeRL 风格 RL 主循环：rollout -> logprob -> reward -> advantage -> update。

    面试讲解重点：
    1. controller 写起来像单进程顺序代码；
    2. 每个 *_fn 在真实系统里可以是远程 WorkerGroup 调用；
    3. DataProto 负责把 response、old_logprob、ref_logprob、reward、advantage
       等字段统一传下去，避免各个 worker 自己约定零散参数；
    4. GRPO/DAPO/OPD 的差异通常落在 reward、advantage 或 actor loss，
       不一定要重写 rollout engine 和分布式训练后端。
    """
    data = MiniDataProto(batch={"prompts": prompts}, meta={"stage": "prompt"})

    rollout = rollout_fn(data)
    data = data.union(rollout)

    old_logprob = actor_logprob_fn(data)
    data = data.union(old_logprob)

    ref_logprob = ref_logprob_fn(data)
    data = data.union(ref_logprob)

    rewards = reward_fn(data)
    data = data.union(rewards)

    advantages = advantage_fn(data)
    data = data.union(advantages)

    metrics = update_actor_fn(data)
    return metrics


def validate_verl_token_alignment(
    data: MiniDataProto,
    fields: Sequence[str] = ("old_logprobs", "ref_logprobs", "advantages", "response_mask"),
) -> Tuple[int, int]:
    """检查 VeRL-style DataProto 中 token 级字段是否对齐。

    面试重点：RL 后训练的 bug 很多不是 loss 公式错，而是 response、logprob、reward、
    advantage、mask 在 token 维度错位。这个函数要求所有字段形状一致，通常是
    [batch, response_len]。
    """
    if not fields:
        raise ValueError("fields must not be empty")
    missing = [name for name in fields if name not in data.batch]
    if missing:
        raise KeyError(f"missing fields: {missing}")

    shape = data.batch[fields[0]].shape
    if len(shape) != 2:
        raise ValueError("token-level fields should be [batch, response_len]")
    for name in fields[1:]:
        if data.batch[name].shape != shape:
            raise ValueError(f"field {name} shape {data.batch[name].shape} != {shape}")
    return int(shape[0]), int(shape[1])


def verl_masked_policy_loss(
    data: MiniDataProto,
    clip_eps: float = 0.2,
    kl_beta: float = 0.04,
) -> torch.Tensor:
    """用 DataProto 字段计算最小版 VeRL token-level actor loss。

    需要字段：
    - new_logprobs：当前 actor 对 rollout token 的 logprob。
    - old_logprobs：rollout 时 policy 的 logprob，是 PPO ratio 的分母。
    - ref_logprobs：reference model 的 logprob，用于 KL 约束。
    - advantages：经过 GAE/GRPO/RLOO/credit assignment 后的 advantage。
    - response_mask：1 表示 assistant 生成 token，0 表示 prompt/pad/tool observation。
    """
    validate_verl_token_alignment(
        data,
        fields=("new_logprobs", "old_logprobs", "ref_logprobs", "advantages", "response_mask"),
    )
    new_logprobs = data.batch["new_logprobs"]
    old_logprobs = data.batch["old_logprobs"]
    ref_logprobs = data.batch["ref_logprobs"]
    advantages = data.batch["advantages"]
    mask = data.batch["response_mask"].to(new_logprobs.dtype)

    ratio = torch.exp(new_logprobs - old_logprobs)
    unclipped = ratio * advantages
    clipped = ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -torch.min(unclipped, clipped)

    # k3 KL: exp(ref-new) - (ref-new) - 1，常用于稳定估计 reference KL。
    ref_delta = ref_logprobs - new_logprobs
    k3_kl = torch.exp(ref_delta) - ref_delta - 1
    total_loss = policy_loss + kl_beta * k3_kl
    return (total_loss * mask).sum() / mask.sum().clamp_min(1.0)

def build_mtp_shifted_labels(labels: torch.Tensor, num_future_tokens: int, ignore_index: int = -100) -> torch.Tensor:
    """
    构造 MTP 的多步 future-token 标签。

    labels: [batch, seq]，通常已经包含 prompt/answer 的 token id 或 ignore_index。
    返回: [batch, num_future_tokens, seq]。

    第 k 个 future head 要在位置 t 预测第 t+k 个 token：
    - k=1 等价于普通 next-token prediction。
    - k=2 表示当前位置额外预测下下个 token。
    - 越靠近序列末尾，越没有足够 future token，所以填 ignore_index。

    面试重点：MTP 不是把一条样本复制多份，而是让同一 hidden state
    同时接收多个未来 token 的监督信号。
    """
    if num_future_tokens <= 0:
        raise ValueError("num_future_tokens must be positive")

    batch, seq_len = labels.shape
    shifted = labels.new_full((batch, num_future_tokens, seq_len), ignore_index)
    for k in range(1, num_future_tokens + 1):
        shifted[:, k - 1, : seq_len - k] = labels[:, k:]
    return shifted


def mtp_multi_head_loss(
    future_logits: torch.Tensor,
    labels: torch.Tensor,
    loss_weights: torch.Tensor | None = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    MTP 多头 future-token cross entropy。

    future_logits: [batch, num_future_tokens, seq, vocab]。
    labels: [batch, seq]。

    最常见的手写理解：
    1. 先把 labels shift 成多个 future target。
    2. 每个 future head 单独计算 CE。
    3. 再用 loss_weights 控制不同预测步长的权重。

    工程提醒：真实大模型里的 MTP head 可能共享或不共享部分参数，
    也可能只训练 MTP 模块，或者 detach 主干 hidden states 来降低扰动。
    """
    if future_logits.ndim != 4:
        raise ValueError("future_logits must be [batch, num_future_tokens, seq, vocab]")

    batch, num_future_tokens, seq_len, vocab_size = future_logits.shape
    targets = build_mtp_shifted_labels(labels, num_future_tokens, ignore_index)

    if loss_weights is None:
        loss_weights = torch.ones(num_future_tokens, device=future_logits.device, dtype=future_logits.dtype)
    loss_weights = loss_weights / loss_weights.sum().clamp_min(1e-12)

    total = torch.zeros((), device=future_logits.device, dtype=future_logits.dtype)
    for k in range(num_future_tokens):
        loss_k = F.cross_entropy(
            future_logits[:, k].reshape(batch * seq_len, vocab_size),
            targets[:, k].reshape(batch * seq_len),
            ignore_index=ignore_index,
        )
        total = total + loss_weights[k] * loss_k
    return total


def speculative_accept_mask(
    draft_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
) -> torch.Tensor:
    """
    极简 speculative decoding 验证：计算草稿 token 的连续接受前缀。

    draft_tokens / target_tokens: [batch, draft_len]。
    返回: [batch, draft_len] bool，True 表示该位置以及之前都匹配。

    这里用 greedy target token 做演示，方便面试手写。
    真正的 lossless speculative decoding 会比较 target/draft 概率，
    并用 rejection sampling 校正被拒绝后的 token 分布。
    """
    if draft_tokens.shape != target_tokens.shape:
        raise ValueError("draft_tokens and target_tokens must have the same shape")
    token_match = draft_tokens == target_tokens
    return token_match.cumprod(dim=-1).bool()


def accepted_prefix_lengths(accept_mask: torch.Tensor) -> torch.Tensor:
    """
    把 speculative accept mask 转成每个样本接受了多少个连续 token。
    """
    return accept_mask.to(torch.long).sum(dim=-1)


@dataclass
class KVBlock:
    """PagedAttention 面试版 block。

    真实 vLLM 会保存每层、每个 head 的 K/V 张量；这里为了能手写和运行，只保存
    token ids，用来解释 block table、物理块复用和碎片控制。
    """

    block_id: int
    tokens: List[int]


class PagedKVCache:
    """极简 Paged KV Cache。

    面试讲法：
    - 传统 KV cache 如果按 sequence 连续分配，长短请求混在一起时容易产生碎片。
    - PagedAttention 把 KV cache 拆成固定大小 block，每个 sequence 维护一张
      block table，把“逻辑 token 位置”映射到“物理 KV block”。
    - 释放一个请求时，只需要回收它占用的物理 block，不必整体搬移其他请求。
    """

    def __init__(self, block_size: int, num_blocks: int):
        if block_size <= 0 or num_blocks <= 0:
            raise ValueError("block_size and num_blocks must be positive")
        self.block_size = block_size
        self.free_blocks = list(range(num_blocks))
        self.blocks: Dict[int, KVBlock] = {}
        self.block_tables: Dict[str, List[int]] = {}

    def _new_block(self) -> int:
        if not self.free_blocks:
            raise RuntimeError("KV cache is full")
        block_id = self.free_blocks.pop(0)
        self.blocks[block_id] = KVBlock(block_id=block_id, tokens=[])
        return block_id

    def allocate_sequence(self, seq_id: str, token_ids: Sequence[int]) -> List[int]:
        """为一个新请求写入 prompt/prefill 阶段的 token。"""
        if seq_id in self.block_tables:
            raise ValueError(f"sequence already exists: {seq_id}")
        self.block_tables[seq_id] = []
        for token_id in token_ids:
            self.append_token(seq_id, int(token_id))
        return self.get_block_table(seq_id)

    def append_token(self, seq_id: str, token_id: int) -> None:
        """decode 阶段追加一个新 token，并在当前 block 满时分配新 block。"""
        if seq_id not in self.block_tables:
            self.block_tables[seq_id] = []
        table = self.block_tables[seq_id]
        if not table or len(self.blocks[table[-1]].tokens) >= self.block_size:
            table.append(self._new_block())
        self.blocks[table[-1]].tokens.append(int(token_id))

    def get_block_table(self, seq_id: str) -> List[int]:
        """返回逻辑 sequence 对应的物理 block id 列表。"""
        return list(self.block_tables.get(seq_id, []))

    def read_tokens(self, seq_id: str) -> List[int]:
        """按 block table 还原 sequence token，用于检查映射是否正确。"""
        tokens: List[int] = []
        for block_id in self.block_tables.get(seq_id, []):
            tokens.extend(self.blocks[block_id].tokens)
        return tokens

    def free_sequence(self, seq_id: str) -> None:
        """释放请求占用的物理 block。真实系统还会处理引用计数和 prefix 共享。"""
        for block_id in self.block_tables.pop(seq_id, []):
            self.blocks.pop(block_id, None)
            self.free_blocks.append(block_id)


def longest_common_prefix(a: Sequence[int], b: Sequence[int]) -> int:
    """计算两个 token 序列的最长公共前缀长度。"""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


@dataclass
class RadixNode:
    """RadixAttention 面试版前缀树节点。"""

    edge: Tuple[int, ...]
    children: Dict[int, "RadixNode"]
    value: str | None = None


class RadixPrefixCache:
    """SGLang / RadixAttention 的极简前缀复用缓存。

    面试讲法：
    - Agent、RAG、多分支采样经常共享 system prompt、工具说明、检索上下文。
    - Radix tree 把公共 token 前缀压缩到同一条路径上，新的请求先查最长命中前缀，
      命中部分可以复用已有 KV cache，只对后缀做 prefill。
    - 这和普通 dict 精确匹配不同：radix cache 关心“最长公共前缀”。
    """

    def __init__(self):
        self.root = RadixNode(edge=(), children={})

    def insert(self, token_ids: Sequence[int], value: str) -> None:
        node = self.root
        remaining = tuple(int(x) for x in token_ids)
        while remaining:
            first = remaining[0]
            child = node.children.get(first)
            if child is None:
                node.children[first] = RadixNode(edge=remaining, children={}, value=value)
                return

            common = longest_common_prefix(remaining, child.edge)
            if common == len(child.edge):
                node = child
                remaining = remaining[common:]
                continue

            # 需要拆边：公共前缀作为中间节点，旧后缀和新后缀变成两个分支。
            common_edge = child.edge[:common]
            old_suffix = child.edge[common:]
            new_suffix = remaining[common:]
            middle = RadixNode(edge=common_edge, children={}, value=None)
            child.edge = old_suffix
            middle.children[old_suffix[0]] = child
            node.children[first] = middle

            if new_suffix:
                middle.children[new_suffix[0]] = RadixNode(edge=new_suffix, children={}, value=value)
            else:
                middle.value = value
            return
        node.value = value

    def match(self, token_ids: Sequence[int]) -> Tuple[int, str | None]:
        """返回最长可复用前缀长度和对应缓存值。"""
        node = self.root
        remaining = tuple(int(x) for x in token_ids)
        matched = 0
        best_value = node.value
        while remaining:
            child = node.children.get(remaining[0])
            if child is None:
                break
            common = longest_common_prefix(remaining, child.edge)
            if common < len(child.edge):
                matched += common
                break
            matched += common
            remaining = remaining[common:]
            node = child
            if node.value is not None:
                best_value = node.value
        return matched, best_value


@dataclass
class DecodeRequest:
    """continuous batching 调度里的一条请求。"""

    request_id: str
    prompt_len: int
    generated_len: int = 0
    max_new_tokens: int = 16


def continuous_batching_step(
    active: Sequence[DecodeRequest],
    waiting: Sequence[DecodeRequest],
    max_batch_size: int,
) -> Tuple[List[DecodeRequest], List[DecodeRequest], List[str]]:
    """极简 continuous batching 调度步骤。

    面试讲法：
    - 静态 batching 会等整批请求全部结束再接新请求，短请求会被长请求拖住。
    - continuous batching 在每个 decode step 后移除完成请求，并从 waiting queue
      补入新请求，让 GPU 尽量保持满载。
    - 真实系统还会考虑 prefill/decode 分离、优先级、KV cache 剩余 block 和超时。
    """
    if max_batch_size <= 0:
        raise ValueError("max_batch_size must be positive")

    next_active: List[DecodeRequest] = []
    finished: List[str] = []
    for req in active:
        updated = DecodeRequest(
            request_id=req.request_id,
            prompt_len=req.prompt_len,
            generated_len=req.generated_len + 1,
            max_new_tokens=req.max_new_tokens,
        )
        if updated.generated_len >= updated.max_new_tokens:
            finished.append(updated.request_id)
        else:
            next_active.append(updated)

    remaining_waiting = list(waiting)
    while len(next_active) < max_batch_size and remaining_waiting:
        next_active.append(remaining_waiting.pop(0))

    scheduled_ids = [req.request_id for req in next_active]
    return next_active, remaining_waiting, scheduled_ids
@dataclass
class ToolResult:
    """一次工具调用结果：工具名 + 文本输出。"""

    name: str
    output: str


def react_loop(
    question: str,
    planner: Callable[[str, Sequence[ToolResult]], Tuple[str, str, Dict[str, str]]],
    tools: Dict[str, Callable[..., str]],
    max_steps: int = 4,
) -> str:
    """
    极简 ReAct loop：Thought -> Action -> Observation。

    面试讲解重点：
    - planner 可以由 LLM 扮演，这里抽象成函数方便手写。
    - unknown tool / tool error 都要进入 observation，而不是直接让程序崩掉。
    - max_steps 是安全阀，防止 Agent 无限循环。
    """
    observations: List[ToolResult] = []
    for _ in range(max_steps):
        thought, tool_name, tool_args = planner(question, observations)
        if tool_name == "final":
            return thought
        if tool_name not in tools:
            observations.append(ToolResult(tool_name, f"unknown tool: {tool_name}"))
            continue
        try:
            observations.append(ToolResult(tool_name, tools[tool_name](**tool_args)))
        except Exception as exc:
            observations.append(ToolResult(tool_name, f"tool error: {exc}"))
    return "Reached max_steps without final answer."

@dataclass
class AgentTraceStep:
    """Agentic RL 轨迹中的一个 step。

    role 用来区分 assistant / tool / user / system。
    只有 assistant 生成的 token 是 policy action，通常参与 RL loss；
    tool observation 是环境返回，应该进入上下文，但不应该被当作模型输出训练。
    """

    role: str
    token_ids: List[int]
    tool_name: str = ""
    is_error: bool = False
    step_reward: float = 0.0
    cost: float = 0.0


def build_agent_response_mask(steps: Sequence[AgentTraceStep]) -> torch.Tensor:
    """根据多轮 Agent trace 构造 response_mask。

    面试抓手：
    - assistant token 记为 1，因为这是策略模型真正采取的动作。
    - tool / environment observation 记为 0，因为这是外部环境返回。
    - mask 一旦错位，PPO/GRPO 的 logprob、ratio、KL 都会跟真实采样路径对不上。
    """
    mask: List[int] = []
    for step in steps:
        is_policy_action = step.role == "assistant"
        mask.extend([1 if is_policy_action else 0] * len(step.token_ids))
    return torch.tensor(mask, dtype=torch.long)


def score_agent_trace(
    steps: Sequence[AgentTraceStep],
    final_success: bool,
    final_reward: float = 1.0,
    error_penalty: float = 0.2,
    tool_cost_weight: float = 0.01,
) -> torch.Tensor:
    """一个可手写的 Agentic RL reward loop 最小版本。

    reward 不是只看最终答案，还可以合并：
    1. final_success：任务是否完成；
    2. step_reward：中间步骤是否有用，例如代码单测部分通过；
    3. error_penalty：工具错误、越权、格式错误的惩罚；
    4. tool_cost_weight：工具调用次数、延迟或费用的成本约束。
    """
    reward = final_reward if final_success else 0.0
    for step in steps:
        reward += step.step_reward
        reward -= tool_cost_weight * step.cost
        if step.is_error:
            reward -= error_penalty
    return torch.tensor(reward, dtype=torch.float32)


def assign_step_credit(
    steps: Sequence[AgentTraceStep],
    final_reward: float,
    gamma: float = 1.0,
    normalize: bool = True,
) -> torch.Tensor:
    """把最终奖励和过程奖励回传到每个 Agent step。

    这是最小版 credit assignment：
    - 从后往前累计 return，让越靠前的 step 也能收到后续成功/失败信号。
    - step.step_reward 可以来自 PRM、代码单测、工具成功率或格式检查。
    - final_reward 代表最终任务是否成功。
    - normalize 对应 GRPO/RLVR 里常见的 advantage 标准化直觉。
    """
    if gamma < 0:
        raise ValueError("gamma must be non-negative")

    returns: List[float] = []
    running_return = final_reward
    for step in reversed(steps):
        running_return = step.step_reward + gamma * running_return
        if step.is_error:
            running_return -= 0.2
        running_return -= 0.01 * step.cost
        returns.append(running_return)
    returns.reverse()

    advantages = torch.tensor(returns, dtype=torch.float32)
    if normalize and advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
    return advantages


def build_token_advantages_from_steps(
    steps: Sequence[AgentTraceStep],
    step_advantages: torch.Tensor,
) -> torch.Tensor:
    """把 step-level advantage 展开到 token 粒度。

    只有 assistant token 会拿到对应 step advantage；tool observation、user/system token
    不是策略动作，advantage 置 0。真实 PPO/GRPO loss 会再配合 response_mask 使用。
    """
    if len(steps) != step_advantages.numel():
        raise ValueError("steps and step_advantages must have the same length")

    token_advantages: List[float] = []
    for step, advantage in zip(steps, step_advantages.tolist()):
        value = advantage if step.role == "assistant" else 0.0
        token_advantages.extend([float(value)] * len(step.token_ids))
    return torch.tensor(token_advantages, dtype=torch.float32)

def replayable_trace_summary(steps: Sequence[AgentTraceStep]) -> List[Dict[str, float | int | str]]:
    """把 trace 压成可审计摘要，方便定位 reward hacking 或 mask 错位。

    真实系统里还会记录 tool input/output、环境版本、随机种子、超时状态等。
    这里保留面试最容易讲清楚的字段。
    """
    summary: List[Dict[str, float | int | str]] = []
    for index, step in enumerate(steps):
        summary.append(
            {
                "step": index,
                "role": step.role,
                "num_tokens": len(step.token_ids),
                "tool_name": step.tool_name,
                "is_error": int(step.is_error),
                "step_reward": float(step.step_reward),
                "cost": float(step.cost),
            }
        )
    return summary

def _quick_check() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 8)
    targets = torch.tensor([1, 3])
    assert torch.allclose(stable_softmax(logits).sum(dim=-1), torch.ones(2))
    assert torch.allclose(cross_entropy_from_logits(logits, targets), F.cross_entropy(logits, targets))

    x = torch.randn(2, 4, 6)
    assert ManualLayerNorm(6)(x).shape == x.shape
    assert ManualRMSNorm(6)(x).shape == x.shape
    assert MultiHeadSelfAttention(6, 2)(x).shape == x.shape
    assert apply_rope(x).shape == x.shape

    rewards = torch.tensor([[1.0, 0.0, 2.0], [0.5, 0.5, 0.5]])
    adv = grpo_group_advantages(rewards)
    assert adv.shape == rewards.shape

    cache = PagedKVCache(block_size=2, num_blocks=4)
    assert cache.allocate_sequence("req-a", [1, 2, 3]) == [0, 1]
    cache.append_token("req-a", 4)
    assert cache.read_tokens("req-a") == [1, 2, 3, 4]
    cache.free_sequence("req-a")
    assert len(cache.free_blocks) == 4

    radix = RadixPrefixCache()
    radix.insert([101, 102, 103, 104], "shared-prefix")
    matched, value = radix.match([101, 102, 999])
    assert matched == 2 and value is None
    radix.insert([101, 102], "short-prefix")
    matched, value = radix.match([101, 102, 999])
    assert matched == 2 and value == "short-prefix"

    active = [DecodeRequest("a", prompt_len=8, generated_len=0, max_new_tokens=1)]
    waiting = [DecodeRequest("b", prompt_len=4, max_new_tokens=2)]
    active, waiting, scheduled = continuous_batching_step(active, waiting, max_batch_size=2)
    assert scheduled == ["b"] and waiting == []

    print("quick check passed")


if __name__ == "__main__":
    _quick_check()




