
import torch
import torch.nn.functional as F
from typing import Tuple
import triton
import triton.language as tl

@torch.no_grad
def zerocot_sampling(
    model,
    input_tokens: torch.LongTensor, # (B, L)
    think_token: int,
    stop_token: int,
    length: int,
) -> Tuple[torch.LongTensor, torch.BoolTensor]:

    model.train(False)
    past_key_values = None
    batch_size = len(input_tokens)
    device = input_tokens.device
    in_thinking = torch.zeros(batch_size, dtype=torch.bool, device=device) # (B,)
    input_pointers = torch.ones((batch_size, 1), dtype=torch.long, device=device) # (B, 1)
    output_tokens = torch.full((batch_size, length+1), fill_value=0, dtype=torch.long, device=device) # (B, L+1)
    output_thinking = torch.zeros((batch_size, length+1), dtype=torch.bool, device=device) # (B, L+1)
    output_tokens[:, 0] = input_tokens[:, 0]
    with torch.no_grad():
        for step in range(length):
            model_outputs = model(
                input_ids=output_tokens[:, step:(step+1)],
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=False,
            )
            # 更新kvcache
            past_key_values = model_outputs.past_key_values
            # 获取out token的logits [B, V]
            out_logits = model_outputs.logits[:, -1, :]
            # 转换为概率分布
            probs = F.softmax(out_logits, dim=-1)
            # 从原始概率分布中采样
            temp_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1) # (B,)
            
            # 是否需要开始思考？
            in_thinking |= (temp_tokens == think_token)

            # 如果在思考，output_tokens取思考的token，且pointer不动；
            # 如果不在思考，取pointer指向的token，然后pointer+1
            output_tokens[:, (step+1)] = torch.where(
                in_thinking,
                temp_tokens,
                torch.gather(input_tokens, dim=1, index=input_pointers).squeeze(1)
            )
            output_thinking[:, (step+1)] = in_thinking

            input_pointers += (~in_thinking).unsqueeze(1)

            # 是否需要结束思考？
            in_thinking &= (temp_tokens != stop_token)
    
    # 生成结果
    return output_tokens, output_thinking


@triton.jit
def truncate_left_align_kernel(
    a_ptr, lengths_ptr, output_ptr,
    B, T, max_len,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offset = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offset < B * T
    b = offset // T
    t = offset % T
    this_t = tl.load(lengths_ptr + b, mask=mask)
    keep_mask = t < this_t
    a_val = tl.load(a_ptr + offset, mask=mask)
    tl.store(output_ptr + b * max_len + t, a_val, mask=mask & keep_mask)

@torch.no_grad
def zerocot_truncate_left_align(a: torch.LongTensor, lengths: torch.LongTensor):
    # a: (B, T), lengths: (B,)
    B, T = a.shape
    max_len = lengths.max().item()
    output = torch.full((B, max_len), -1, dtype=a.dtype, device=a.device)
    BLOCK_SIZE = 128
    grid = lambda _: (triton.cdiv(B * T, BLOCK_SIZE),)
    assert a.is_contiguous()
    assert lengths.is_contiguous()
    assert output.is_contiguous()
    truncate_left_align_kernel[grid](
        a, lengths, output,
        B, T, max_len,
        BLOCK_SIZE=BLOCK_SIZE
    )
    mask = output >= 0
    actual_output = torch.clamp(output, min=0)
    return actual_output, mask

@triton.jit
def fill_address_kernel(
    read_mask_ptr, prefix_sum_m1_ptr, address_out_ptr, # (B, T), (B, T), (B, T0)
    B, T, T0,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offset = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offset < B * T
    b = offset // T
    t = offset % T
    read_mask = tl.load(read_mask_ptr + offset, mask=mask)
    idx = tl.load(prefix_sum_m1_ptr + offset, mask=mask)
    tl.store(address_out_ptr + b * T0 + idx, t, mask=mask & read_mask)

@triton.jit
def fill_logprob_kernel(
    logprob_in_ptr, address_ptr, logprob_out_ptr, # (B, T0), (B, T0), (B, T)
    B, T0, T,
    BLOCK_SIZE: tl.constexpr
):
    # address: up to T
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offset = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offset < B * T0
    b = offset // T0
    address = tl.load(address_ptr + offset, mask=mask)
    address_mask = address >= 0
    logprob_in = tl.load(logprob_in_ptr + offset, mask=mask)
    tl.store(logprob_out_ptr + b * T + address, logprob_in, mask=mask & address_mask)


@torch.no_grad
def zerocot_reward_fill(
    base_nll_bt0: torch.FloatTensor, # (B, T0)
    actor_nll_bt: torch.FloatTensor, # (B, T)
    actor_read_mask_bt: torch.BoolTensor, # (B, T)
    thinking_default_reward: float = 0.0,
):
    # reward = base_nll - actor_nll
    # 1. moveto -> base_nll_bt
    # 2. compute reward
    device = base_nll_bt0.device
    B, T0 = base_nll_bt0.shape
    B, T = actor_nll_bt.shape
    BLOCK_SIZE = 128

    base_nll_bt = torch.clone(actor_nll_bt)
    prefix_sum_m1_bt = torch.cumsum(actor_read_mask_bt, dim=1) - 1
    address_bt0 = -torch.ones((B, T0), dtype=torch.long, device=device)
    grid_bt = lambda _: (triton.cdiv(B * T, BLOCK_SIZE),)
    fill_address_kernel[grid_bt](
        actor_read_mask_bt, prefix_sum_m1_bt, address_bt0,
        B, T, T0,
        BLOCK_SIZE=BLOCK_SIZE
    )
    grid_bt0 = lambda _: (triton.cdiv(B * T0, BLOCK_SIZE),)
    fill_logprob_kernel[grid_bt0](
        base_nll_bt0, address_bt0, base_nll_bt,
        B, T0, T,
        BLOCK_SIZE=BLOCK_SIZE
    )
    reward_bt = torch.where(actor_read_mask_bt, base_nll_bt - actor_nll_bt, thinking_default_reward)
    return reward_bt

@torch.no_grad
def zerocot_discounted_cumsum(a: torch.tensor, gamma: float):
    i = 1
    b = a.clone()
    B, length = a.shape
    while i < length:
        b[:, :length-i] += gamma * b[:, i:]
        i *= 2
        gamma **= 2
    return b


@torch.compile
def zerocot_compute_nll(
    logits: torch.FloatTensor, # (B, T, V)
    labels: torch.LongTensor, # (B, T)
    reading_mask: torch.BoolTensor, # (B, T)
    think_token: int,
    stop_token: int,
):
    masked_logits = logits.clone()
    masked_logits[:, :, think_token].masked_fill_(reading_mask, -1e20)
    masked_logits[:, :, stop_token].masked_fill_(reading_mask, -1e20)

    log_sum_exp_all = torch.logsumexp(logits, dim=-1) # (B, T)
    log_sum_exp_mask = torch.logsumexp(masked_logits, dim=-1) # (B, T)
    selected_token_logits = torch.gather(logits, 2, labels.unsqueeze(-1)).squeeze(-1)

    no_think_when_read_nll   = log_sum_exp_all  - log_sum_exp_mask
    token_given_no_think_nll = log_sum_exp_mask - selected_token_logits
    token_when_think_nll     = log_sum_exp_all  - selected_token_logits

    return no_think_when_read_nll, token_given_no_think_nll, token_when_think_nll
