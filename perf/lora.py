import torch

# Hello world Low Rank Adaptation

class LinearLoRA(torch.nn.Module):
    def __init__(self, rank, alpha):
        super().__init__()
        (in_dim, out_dim) = (32, 64)
        
        # pretend this is pretrained weight, say MLP/MoE or attention
        self.original_weight = torch.nn.Linear(in_dim, out_dim, False)

        self.original_weight.weight.requires_grad = False

        self.adapter_in =  torch.nn.Linear(in_dim, rank, False)
        self.adapter_out =  torch.nn.Linear(rank, out_dim, False)

        # https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
        torch.nn.init.kaiming_uniform_(self.adapter_in.weight, a=5**0.5)
        torch.nn.init.zeros_(self.adapter_out.weight)

        self.ratio = alpha / rank

    def forward(self, x):
        return self.original_weight(x) + self.ratio * self.adapter_out(self.adapter_in(x))
    
if __name__ == '__main__':
    rank = 4
    alpha = 1
    model = LinearLoRA(rank, alpha)

    for name, param in model.named_parameters():
        print(name, param.requires_grad)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(10):
        optimizer.zero_grad()
        batch = torch.rand(16, 32)
        loss: torch.Tensor = model(batch).mean()
        loss.backward()
        optimizer.step()
        print(f'Step {step} | loss {loss.item()}')

