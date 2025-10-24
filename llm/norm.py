import torch


class RMSNorm(torch.nn.Module):
    def __init__(self, dim_model, epsilon=1e-5):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.rand(dim_model))
        self.dim_model = dim_model
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor):
        rms = torch.sum(x*x, dim=-1).unsqueeze(-1) / self.dim_model + self.epsilon
        rms = rms.sqrt()
        return x / rms * self.gain 

if __name__ == '__main__':
    batch = 16
    seqlen = 32
    d_model = 64

    x = torch.rand(batch, seqlen, d_model)
     
    norm = RMSNorm(d_model)
    y = norm(x)
    print(y.shape)