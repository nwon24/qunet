import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.io import decode_image
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from torch.nn.functional import one_hot
from qconv2d import QConv2d

class kits19Dataset(Dataset):
    def __init__(self, imgdir, segdir):
        self.imgdir = imgdir
        self.segdir = segdir
        self.imglist = [f for f in Path(imgdir).iterdir() if f.is_file()]
        self.seglist = [f for f in Path(segdir).iterdir() if f.is_file()]
        #self.img = [decode_image(file).to(torch.float16) for file in self.imglist]
        #self.seg = [decode_image(file).to(torch.float16) for file in self.seglist]

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, idx):
        img =decode_image(self.imglist[idx]).to(torch.float32)
        seg = decode_image(self.seglist[idx], mode="GRAY").to(torch.float32)
        seg = torch.squeeze(((seg / 255) * 2).to(torch.int64))
        #print(seg)
        #print(torch.unique(seg))
        #seg = torch.squeeze(one_hot(seg, 3).to(torch.float32))
        #print(seg.shape)
        return img, seg
        #return self.img[idx], self.seg[idx]

class Lambda(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)

class UNet(nn.Module):
    def __init__(self, width, in_chan, kernel_size=3, padding=1, n_classes=3):
        super().__init__()
        self.width = width
        self.kernel_size = kernel_size
        self.padding = padding
        self.encoder = nn.Sequential(
                QConv2d(in_chan, 64, kernel_size=kernel_size, padding=padding), # 0
                nn.ReLU(),                                                          
                QConv2d(64, 64, kernel_size=kernel_size, padding=padding),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                QConv2d(64, 128, kernel_size=kernel_size, padding=padding),
                nn.ReLU(),
                QConv2d(128, 128, kernel_size=kernel_size, padding=padding),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                QConv2d(128, 256, kernel_size=kernel_size, padding=padding),
                nn.ReLU(),
                QConv2d(256, 256, kernel_size=kernel_size, padding=padding),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                QConv2d(256, 512, kernel_size=kernel_size, padding=padding),
                nn.ReLU(),
                QConv2d(512, 512, kernel_size=kernel_size, padding=padding),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                QConv2d(512, 1024, kernel_size=kernel_size, padding=padding),
                nn.ReLU(),
                QConv2d(1024, 1024, kernel_size=kernel_size, padding=padding),
                nn.ReLU(),
                )
        self.outputs = {}
        def get_activation(name):
            def hook(model, input, output):
                self.outputs[name] = output.detach()
            return hook
        for i in range(3, len(self.encoder), 5):
            self.encoder[i].register_forward_hook(get_activation("el%d" % (i)))

        self.decoder = nn.Sequential(
                nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
                Lambda(lambda x: torch.cat([x, self.outputs["el18"]], dim=1)),
                QConv2d(1024, 512, kernel_size=kernel_size, padding=padding),
                nn.ReLU(),
                QConv2d(512, 512, kernel_size=kernel_size, padding=padding),
                nn.ReLU(),
                nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
                Lambda(lambda x: torch.cat([x, self.outputs["el13"]], dim=1)),
                QConv2d(512, 256, kernel_size=kernel_size, padding=padding),
                nn.ReLU(),
                QConv2d(256, 256, kernel_size=kernel_size, padding=padding),
                nn.ReLU(),
                nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
                Lambda(lambda x: torch.cat([x, self.outputs["el8"]], dim=1)),
                QConv2d(256, 128, kernel_size=kernel_size, padding=padding),
                nn.ReLU(),
                QConv2d(128, 128, kernel_size=kernel_size, padding=padding),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
                Lambda(lambda x: torch.cat([x, self.outputs["el3"]], dim=1)),
                QConv2d(128, 64, kernel_size=kernel_size, padding=padding),
                nn.ReLU(),
                QConv2d(64, 64, kernel_size=kernel_size, padding=padding),
                nn.ReLU(),
                QConv2d(64, n_classes, kernel_size=1),
                )

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)
