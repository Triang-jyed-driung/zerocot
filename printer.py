import torch

def print0(*message, **kwargs):
    """If distributed is initialized print only on rank 0."""
    kwargs['flush'] = True
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return
    print(*message, **kwargs)
