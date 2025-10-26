import torch
from jaxtyping import Float


def gspo_loss_fn(
    current_policy_logprob: Float[torch.Tensor, 'batch group sequence'], 
    old_policy_logprob: Float[torch.Tensor, 'batch group sequence'], 
    rewards: Float[torch.Tensor, 'batch group'],
    epsilon: float
):  
    imporance_ratio = (current_policy_logprob - old_policy_logprob).mean(dim=-1).exp()
    advantage = (rewards - rewards.mean(dim=-1, keepdim=True)) / rewards.std(dim=-1, keepdim=True)

    term1 = imporance_ratio * advantage
    term2 = torch.clamp(imporance_ratio, min=1-epsilon, max=1+epsilon) * advantage
    gspo_objective = torch.min(term1, term2).mean()

    return -gspo_objective

if __name__ == '__main__':
    size = (batch, group, sequence) = (4, 16, 128)
    current_policy_logprob = torch.rand(size)
    old_policy_logprob = torch.rand(size)
    rewards = torch.rand(batch, group)
    epsilon = 0.1
    loss = gspo_loss_fn(current_policy_logprob, old_policy_logprob, rewards, epsilon)
    print(loss)