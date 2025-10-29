import torch
from jaxtyping import Float


def temperature_scaled_probabilities(llm_logits: Float[torch.Tensor, 'batch sequence vocab'], temperature: float):
    next_word_prob = llm_logits[:, -1, :] / temperature
    next_word_prob = next_word_prob.softmax(dim=-1)
    return next_word_prob

def nucleus_sampling():
    pass

class KVCache:
    pass

if __name__ == '__main__':
    size = (batch, sequence, vocab) = (4, 8, 16)
    llm_logits = torch.rand(size)
    temp = 0.5
    prob = temperature_scaled_probabilities(llm_logits, temp)
    print(prob.shape)