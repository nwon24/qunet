import torch
import torch.nn as nn
import imageio
from unet import  kits19Dataset, UNet
from torchvision.io import decode_image, write_jpeg
from torchvision.utils import save_image
import sys
import pandas as pd
import numpy as np

if len(sys.argv) < 4:
    print("Invalid arguments")
    quit()

model_file = sys.argv[1]
target_image = sys.argv[2]
target_seg = sys.argv[3]

model = UNet(512, 1)
model.load_state_dict(torch.load(sys.argv[1], weights_only=True))
model.eval()

target_image_tensor = decode_image(sys.argv[2]).to(torch.float32).unsqueeze(0)
target_seg_tensor= decode_image(target_seg, mode="GRAY").to(torch.float32)
target_seg_tensor= torch.squeeze(((target_seg_tensor / 255) * 2).to(torch.int64))
#output = nn.Softmax2d()(model(target_image_tensor).squeeze(0))
output = model(target_image_tensor)
print(nn.CrossEntropyLoss()(output, target_seg_tensor.unsqueeze(0)))
print(target_seg_tensor)
output=nn.Softmax2d()(output).squeeze(0)
print(torch.unique(output[0]))
print(torch.unique(output[1]))
print(torch.unique(output[2]))
#print(torch.unique(torch.max(output, dim=0).indices))
print(torch.unique(torch.max(output[0])), torch.unique(torch.min(output[0])))
print(torch.unique(torch.max(output[1])), torch.unique(torch.min(output[1])))
print(torch.unique(torch.max(output[2])), torch.unique(torch.min(output[2])))
#print(torch.mean(output[0]))
#print(torch.mean(output[1]))
#print(torch.mean(output[2]))
output_seg=torch.zeros_like(target_seg_tensor.squeeze())
output_seg.requires_grad = False
print(output_seg.shape)
print(output.shape)
#output_seg[output[1] > 0.2] = 1
#output_seg[output[2] > 0.5] = 2
output_seg = torch.argmax(output, dim = 0)
output_seg = output_seg * 255 /2 
output_seg = output_seg.to(torch.uint8)
imageio.imwrite("output.jpg", output_seg) 
print(torch.sum(target_seg_tensor==0))
print(torch.sum(target_seg_tensor==1))
print(torch.sum(target_seg_tensor==2))
