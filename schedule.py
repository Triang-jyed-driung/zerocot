import math

# 自定义调度函数
def cosine_lr_lambda(current_step, total_steps, warmup_steps, lr, lr_end):
    if current_step < warmup_steps and warmup_steps > 0:
        return current_step / warmup_steps
    # progress <= 1
    progress = min(1, (current_step - warmup_steps) / max(1, total_steps - warmup_steps))
    return (lr_end + (lr - lr_end) * 0.5 * (1.0 + math.cos(math.pi * progress))) / lr 
