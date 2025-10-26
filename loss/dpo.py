from typing import Tuple

import torch
from jaxtyping import Float


def dpo_loss_and_reward_fn(
    current_policy_logp: Float[torch.Tensor, 'batch 2'],
    ref_policy_logp: Float[torch.Tensor, 'batch 2'], 
    beta: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """In the last dimension of logp, we have the pair (positive, negative) of samples.
    Returns loss and reward.
    """

    term_pos = current_policy_logp[..., 0] - ref_policy_logp[..., 0]
    term_neg = current_policy_logp[..., 1] - ref_policy_logp[..., 1]

    loss = - torch.nn.functional.logsigmoid(beta * (term_pos - term_neg)).mean()
    rewards = beta * (current_policy_logp - ref_policy_logp)
    return loss, rewards


if __name__ == '__main__':
    batch = 4
    seq_len = 32    # assuming const sequence length (padded if necessary)
    current_policy_logp = torch.rand(batch, 2)
    ref_policy_logp = torch.rand(batch, 2)
    beta = 0.1
    loss, reward = dpo_loss_and_reward_fn(current_policy_logp, ref_policy_logp, beta)
    print(loss)