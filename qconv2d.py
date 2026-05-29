from functools import partial
import torch
import torch.nn as nn
import pennylane as qml

class QConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.wires = kernel_size**2
        self.dev = qml.device("lightning.qubit", wires=self.wires)
        self.pad, self.stride = padding, stride
        self.filter_weights = nn.Parameter(torch.rand(out_channels, in_channels, kernel_size, kernel_size))
        self.filter_biases = nn.Parameter(torch.rand(out_channels, ))
        @partial(qml.batch_input, argnum=0)
        @qml.qnode(self.dev)
        def qc(inputs, weights):
            #print(inputs.shape)
            wires = self.kernel_size*self.kernel_size
            weights = weights.flatten()
            inputs=inputs.flatten(start_dim=1, end_dim=-1)
            qml.AngleEmbedding(inputs, wires=range(wires))
            for i in range(wires):
                qml.CNOT(wires=[i,(i+1)%wires])
            for i in range(wires):
                qml.RY(weights[i], wires=i)
            return qml.expval(qml.Z(0))

        self.qc = qc


    def forward(self, X, padding_value=0):
        N, C, H,  W  = X.shape
        F, C, HH, WW = self.filter_weights.shape
        X_pad = nn.functional.pad(X, (self.pad, self.pad, self.pad, self.pad), mode='constant', value=padding_value)
        _, _, H_pad, W_pad = X_pad.shape
        H_out = 1 + (H + 2*self.pad - HH) // self.stride
        W_out = 1 + (W + 2*self.pad - WW) // self.stride
        out = torch.zeros((N,F,H_out,W_out))
        for n in range(N):
            print(f"processing batch_number {n}")
            for f in range(F):
                print(f"\tprocessing filter {f}")
                for h in range(H_out):
                    print(f"\t\tprocessing height {h}")
                    for w in range(W_out):
                        if w % 64 == 0:
                            print(f"\t\t\tprocessing width {w}")
                        h_, w_ = h*self.stride, w*self.stride
                        patches = X_pad[:, :, h_:h_+HH, w_:w_+WW].squeeze()
                        output = self.qc(patches, self.filter_weights[f].unsqueeze(0))
#                        print(output.shape)
                        out[:, f, h, w] = output

        return out

