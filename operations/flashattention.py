import torch


class FlashAttention2(torch.autograd.Function):
    
    # TODO

    @staticmethod
    def forward(ctx, x, weight):
        pass
    
    @staticmethod
    def backward(ctx, grad_out):
        pass

if __name__ == '__main__':
    fa2 = FlashAttention2.apply