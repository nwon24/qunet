from functools import partial
import torch
import torch.nn as nn
import pennylane as qml
import torch.nn.functional as Func
from torch.distributions import Normal

class QConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.wires = kernel_size**2
        self.dev = qml.device("lightning.gpu", wires=self.wires)
        self.pad, self.stride = padding, stride
        self.filter_weights = nn.Parameter(torch.rand(out_channels, in_channels, kernel_size, kernel_size))
        self.filter_biases = nn.Parameter(torch.rand(out_channels, ))
        self.mu = nn.Parameter(torch.tensor([0.0]))
        self.sigma = nn.Parameter(torch.tensor([0.0]))
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        @partial(qml.batch_params)
        @qml.qnode(self.dev, diff_method="adjoint", interface='torch')
        def qc(inputs, weights):

            qml.AngleEmbedding(inputs, wires=range(self.wires))
            
            qml.BasicEntanglerLayers(weights, wires=range(self.wires))
            return qml.expval(qml.Z(0))

        self.qc = qc
        


    def forward(self, X, padding_value=0):
        #print(f"X.shape in forward {X.shape}")
        #print(X.requires_grad)
        N, C, H,  W  = X.shape
        F, C, HH, WW = self.filter_weights.shape
        X_pad = nn.functional.pad(X, (self.pad, self.pad, self.pad, self.pad), mode='constant', value=padding_value)
        _, _, H_pad, W_pad = X_pad.shape
        H_out = 1 + (H + 2*self.pad - HH) // self.stride
        W_out = 1 + (W + 2*self.pad - WW) // self.stride

        unfolded = nn.functional.unfold(X, kernel_size=self.kernel_size,padding=self.pad,stride=self.stride)
        squashed_weights=self.filter_weights.view(F,-1)

        z = torch.normal(torch.zeros(N,self.wires),torch.ones(N,self.wires)).to(self.mu.device.type)
        weights=((z+self.mu)*torch.exp(self.sigma)).unsqueeze(1)
        m = torch.cat(
             [self.qc(unfolded[:,:,i].view(N, C, self.kernel_size * self.kernel_size),
                      weights) for i in range(unfolded.shape[-1])],
             dim=-1)

        return m.view(N, F, H_out, W_out)



