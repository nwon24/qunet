import os
from unet import kits19Dataset, UNet
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import random_split, DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def train(device, model, dataloader, lossfn, optim, epochs):
    model.train()
    losses = np.zeros(epochs)
    numbatches = len(dataloader)
    for epoch in range(epochs):
        totalloss = 0;
        for (x,y) in dataloader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = lossfn(pred, y)
            loss.backward()
            optim.step()
            optim.zero_grad()

            totalloss += loss.item()
        avgloss = totalloss / numbatches
        losses[epoch] = avgloss
        print("Epoch {epoch}: average loss {avgloss:.5f}")
    return losses

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    acc = torch.accelerator.current_accelerator()
    backend = dist.get_default_backend_for_device(acc)
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def run(fn, world_size):
    mp.spawn(fn, args=(world_size,), nprocs=world_size, join=True)

def prepare(rank, world_size, batch_size, dataset):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                 shuffle=False, drop_last=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                            pin_memory=False, drop_last=False,
                            shuffle=False, sampler=sampler)
    return dataloader

def main(rank, world_size):
    train_size = 80000
    batch_size = 64
    lr=1e-3
    epochs = 50

    setup(rank, world_size)
    data = kits19Dataset("image", "seg")
    train_dataset, test_dataset = random_split(data, [train_size, len(data) - train_size])
    train_dataloader = prepare(rank, world_size, batch_size, train_dataset)
    test_dataloader = prepare(rank, world_size, batch_size, test_dataset)
    model = UNet(512, 1).to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank, 
                find_unused_parameters=True)

    lossfn = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr=lr)

    train_losses = train(rank, model, train_dataloader, lossfn, optim, epochs)
    np.savetxt("trainlosses.csv", train_losses, delimiter=",", fmt="%f")

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    run(main, world_size)
