import os
import sys
from qunet import kits19Dataset
from qunet import UNet
#from qunet_lite import UNet
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import random_split, DataLoader, SubsetRandomSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from sklearn.model_selection import train_test_split

def train(device, model, dataloader, lossfn, optim, epochs_completed, epochs):
    model.train()
    losses = np.zeros(epochs)
    numbatches = len(dataloader)
    #print(f"Number of batches: {numbatches}")
    for epoch in range(epochs_completed+1, epochs):
        totalloss = 0
        for (x,y) in dataloader:
            x = x.to(device)
            y = y.to(device)
            #print("here!")
            optim.zero_grad(set_to_none=True)
            pred = model(x)
            loss = lossfn(pred, y)
            # print(loss.item())
            loss.backward()
            optim.step()
            #print(f"Batch loss: {loss.item()}")
            totalloss += loss.item()

        avgloss = totalloss / numbatches
        losses[epoch] = avgloss
        print(f"Epoch {epoch}: average loss {avgloss:.5f}")
        torch.save(model.state_dict(), f"QUNET_e{epoch}.pth")
    return losses

def main():
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    batch_size = 14
    lr=1e-3
    epochs = 50
    epochs_completed = 0


    data = kits19Dataset("size_64/image", "size_64/seg")
    lencap = 1000
    train_size = int(len(data) * 0.8)
    test_size = len(data) - train_size
    torch.Generator().manual_seed(1234)
    train_dataset, test_dataset = random_split(data, [train_size, test_size])
    torch.seed()
    subset_size = 8*batch_size
    indices = torch.randperm(len(train_dataset))[:subset_size]
    sampler = SubsetRandomSampler(indices)
    #train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    model = UNet(64, 1).to(device)
    if len(sys.argv) > 1:
        model.load_state_dict(torch.load(sys.argv[1], weights_only=True))
        epochs_completed = int(sys.argv[1][-5])

    lossfn = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr=lr)

    train_losses = train("cuda", model, train_dataloader, lossfn, optim, epochs_completed, epochs)
    np.savetxt("trainlosses.csv", train_losses, delimiter=",", fmt="%f")

if __name__ == "__main__":
    main()
