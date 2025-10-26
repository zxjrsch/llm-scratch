import torch
from jaxtyping import Float


def cross_entropy_loss_fn(llm_unembedding: Float[torch.Tensor, 'batch sequence vocab'], tokens: Float[torch.Tensor, 'batch sequence']):
    """Equivalently, the negative log-likelihood. The llm_unembedding is the LLM's pre-softmax output (aka logits), we use this and
    not directly softmax in order to cancel out log and exp whenever possible for numerical stability. Moreover since softmax is 
    translational invarant softmax(X + C) = softmax(X) we will subtract each component of X by max(X1, ..., Xn) for numerical stability.
    """
    llm_unembedding -= llm_unembedding.max(dim=-1, keepdim=True).values
    token_level_loss = torch.logsumexp(llm_unembedding, dim=-1) - torch.gather(input=llm_unembedding, dim=-1, index=tokens.unsqueeze(-1)).squeeze(-1)
    return token_level_loss.mean()  # average across batch and sequence


if __name__ == '__main__':
    batch = 32
    sequence = 32
    vocab_size = 201088

    llm_unembedding = torch.rand(batch, sequence, vocab_size)
    tokens = torch.randint(low=0, high=vocab_size-1, size=(batch, sequence))
    loss = cross_entropy_loss_fn(llm_unembedding, tokens)
    print(loss)