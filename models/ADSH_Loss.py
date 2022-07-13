import torch.nn as nn
import torch

class ADSH_Loss(nn.Module):
    def __init__(self, code_length, gamma):
        super(ADSH_Loss, self).__init__()
        self.code_length = code_length
        self.gamma = gamma

    def forward(self, F, B, S, omega):
        hash_loss = ((self.code_length * S - F @ B.t()) ** 2).sum() / (F.shape[0] * B.shape[0]) / self.code_length * 12
        quantization_loss = ((F - B[omega, :]) ** 2).sum() / (F.shape[0] * B.shape[0]) * self.gamma / self.code_length * 12

        loss = hash_loss + quantization_loss
        return loss, hash_loss, quantization_loss