from functools import partial
import torch
import torch.nn as nn
import pennylane as qml
import torch.nn.functional as Func

class QConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.wires = kernel_size**2
        self.dev = qml.device("lightning.gpu", wires=self.wires)
        self.pad, self.stride = padding, stride
        self.filter_weights = nn.Parameter(torch.rand(out_channels, in_channels, kernel_size, kernel_size))
        self.filter_biases = nn.Parameter(torch.rand(out_channels, ))
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        #@partial(qml.batch_input, argnum=0)
        #@qml.qjit
        #@partial(qml.batch_input, argnum=0)
        #@partial(qml.batch_input, argnum=0)
        @partial(qml.batch_params)
        #@qml.batch_params
        #@qml.batch_partial
        @qml.qnode(self.dev, diff_method="adjoint", interface='torch')
        def qc(inputs, weights):
        #def qc(inputs, weights):
            # print(f"Inputs shape in qc {inputs.shape}")
            #weights = weights.reshape(-1)
            #print(inputs.shape)
            #print(weights.shape)

            qml.AngleEmbedding(inputs, wires=range(self.wires))
            
            # for i in range(self.wires):
            #     qml.CNOT(wires=[i, (i + 1) % self.wires])

            # for i in range(self.wires):
            #     qml.RY(weights[i], wires=i)
                #qml.RY(1, wires=i)
            qml.BasicEntanglerLayers(weights, wires=range(self.wires))
            return qml.expval(qml.Z(0))

        self.qc = qc
        #self.qc = qml.simplify(self.qc)
        


    def forward(self, X, padding_value=0):
        #print(f"X.shape in forward {X.shape}")
        #print(X.requires_grad)
        N, C, H,  W  = X.shape
        F, C, HH, WW = self.filter_weights.shape
        X_pad = nn.functional.pad(X, (self.pad, self.pad, self.pad, self.pad), mode='constant', value=padding_value)
        _, _, H_pad, W_pad = X_pad.shape
        H_out = 1 + (H + 2*self.pad - HH) // self.stride
        W_out = 1 + (W + 2*self.pad - WW) // self.stride
        #print(self.filter_weights.shape)
        #print(m.shape)
        #print(H_out, W_out)
        unfolded = nn.functional.unfold(X, kernel_size=self.kernel_size,padding=self.pad,stride=self.stride)
        squashed_weights=self.filter_weights.view(F,-1)
        #squashed_weights=self.filter_weights.view(F,C, self.kernel_size*self.kernel_size)
        #print(unfolded.shape)
        #print(squashed_weights.shape)
        #unfolded.requires_grad = False
        #print(unfolded.requires_grad)
        # m = torch.cat(
        #     [self.qc(unfolded[:,:,i].view(N, C, self.kernel_size * self.kernel_size), self.filter_weights.view(-1, self.kernel_size * self.kernel_size).unsqueeze(0).repeat(N, 1, 1)) for i in range(unfolded.shape[-1])],
        #     dim=-1)
        m = torch.cat(
            [self.qc(unfolded[:,:,i].view(N, C, self.kernel_size * self.kernel_size),
                     torch.normal(torch.zeros(N, self.wires),torch.ones(N, self.wires))) for i in range(unfolded.shape[-1])],
            dim=-1)
        #print(m)
        #print(m.shape)
        #self.qc(unfolded[:,:,0].view(N, C, self.kernel_size * self.kernel_size), squashed_weights)
        #return (squashed_weights @ unfolded).view(N, F, H_out, W_out)
        #m.requires_grad = True
        return m.view(N, F, H_out, W_out)

    # def forward(self, X, padding_value=0):
    #     # print(f"X.shape in forward {X.shape}")
    #     N, C, H,  W  = X.shape
    #     F, C, HH, WW = self.filter_weights.shape
    #     X_pad = nn.functional.pad(X, (self.pad, self.pad, self.pad, self.pad), mode='constant', value=padding_value)
    #     _, _, H_pad, W_pad = X_pad.shape
    #     H_out = 1 + (H + 2*self.pad - self.kernel_size) // self.stride
    #     W_out = 1 + (W + 2*self.pad - self.kernel_size) // self.stride
    #     L = H_out * W_out
    #     # out = torch.zeros((N,F,H_out,W_out))

    #     patches = Func.unfold(
    #         X,
    #         kernel_size=self.kernel_size,
    #         padding=self.pad,
    #         stride=self.stride
    #     )

    #     # Reshape to separate channels and patch pixels.
    #     # [N, C*K*K, L] -> [N, C, K*K, L]
    #     patches = patches.view(N, C, self.kernel_size ** 2, L)

    #     # Move spatial patch index beside batch.
    #     # [N, C, K*K, L] -> [N, L, C, K*K]
    #     patches = patches.permute(0, 3, 1, 2)

    #     outputs = []

    #     for f in range(self.out_channels):
    #         # Accumulate over input channels.
    #         acc = X.new_zeros(N, L)

    #         for c in range(C):
    #             # [N, L, K*K]
    #             patch_fc = patches[:, :, c, :]

    #             # [N, L, K*K] -> [N*L, K*K]
    #             patch_fc = patch_fc.reshape(N * L, self.kernel_size ** 2)

    #             # Quantum output: [N*L]
    #             qout = self.qc(patch_fc, self.filter_weights[f, c])

    #             # [N*L] -> [N, L]
    #             qout = qout.reshape(N, L)

    #             acc = acc + qout

    #         acc = acc + self.filter_biases[f]
    #         outputs.append(acc)

    #     # [F_out, N, L] -> [N, F_out, L]
    #     out = torch.stack(outputs, dim=1)

    #     # [N, F_out, L] -> [N, F_out, H_out, W_out]
    #     out = out.view(N, self.out_channels, H_out, W_out)

    #     return out

