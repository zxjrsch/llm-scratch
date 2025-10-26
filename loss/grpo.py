import torch
from einops import rearrange
from jaxtyping import Float


# Any policy has shape [questions, group, response_sequence]
# A reward is assigned to each element of [questions, group]
# Assume we receive softmax probability
def grpo_loss_fn(ref_policy: torch.Tensor, old_policy: torch.Tensor, current_policy: torch.Tensor, rewards: torch.Tensor, epsilon: float, beta: float):
    advantage: Float[torch.Tensor, 'question group'] = (rewards - rewards.mean(dim=-1, keepdim=True)) / (rewards.std(dim=-1, keepdim=True) + 1e-8)
    advantage = rearrange(advantage, 'q g -> q g 1')
    ratio = current_policy / old_policy
    t1 = advantage * ratio
    t2 = advantage * torch.clamp(ratio, min=1-epsilon, max=1+epsilon)

    # KL(current || ref)
    ratio = ref_policy / current_policy 
    kl = ratio - ratio.log() - 1

    grpo_loss = torch.max(t1, t2) - beta * kl
    # average over tokens, groups, and questions
    grpo_loss = grpo_loss.mean()
    # the GRPO loss is to be maximized, thus we return its negative 
    # to be minimized by gradient descent
    return -grpo_loss

    
if __name__ == '__main__':
    torch.manual_seed(0)
    batch_size = (questions, group, out_sequence) = (5, 10, 15)
    ref_policy = torch.rand(batch_size)
    old_policy = torch.rand(batch_size)
    current_policy = torch.rand(batch_size)
    rewards = torch.rand(questions, group)
    epsilon = 1
    beta = 1
    loss = grpo_loss_fn(ref_policy, old_policy, current_policy, rewards, epsilon, beta)
    print(loss)