import math
from collections.abc import Callable, Iterable
from typing import Optional

import torch


def lr_scheduler(current_step, num_warmup_steps, num_cos_annealing_steps, lr_min, lr_max):
    assert current_step >= 0
    if current_step < num_warmup_steps:
        return current_step / num_warmup_steps * lr_max
    elif current_step < (num_warmup_steps + num_cos_annealing_steps):
        arg = (current_step - num_warmup_steps) / num_cos_annealing_steps * math.pi
        return lr_min + (lr_max - lr_min)/2 * (1 + math.cos(arg))
    return lr_min

# TODO implement a distributed optimizer, see ZeroRedundancyOptimizer and https://docs.pytorch.org/docs/2.9/distributed.optim.html

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        """params: trainable model parameters"""
        if lr <=0:
            raise ValueError(f'Learning rate must be positive.')
        defaults = {
            'lr': lr
        }
        super().__init__(params=params, defaults=defaults)
    
    def step(self, closure: Optional[Callable]=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get('t', 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t+1) * grad
                state['t'] = t + 1

        return loss


class Muon(torch.optim.Optimizer):
    # Kimi K2 https://arxiv.org/pdf/2507.20534
    # https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py
    # https://github.com/karpathy/nanochat/blob/master/nanochat/muon.py

    pass

class AdamW(torch.optim.Optimizer):
    pass

if __name__ == '__main__':
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=1)

    for t in range(100):
        opt.zero_grad()
        loss = weights.mean()
        print(f'Step {t} loss {loss.cpu().item()}')
        loss.backward()
        opt.step()