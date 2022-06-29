from topnet import TopNet18
import wandb

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import transforms
import pandas as pd
from pathlib import Path
import pickle

from tqdm import tqdm
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR100
from torch import nn
from torch_ema import ExponentialMovingAverage

import os
# For logging purposes
os.environ['WANDB_API_KEY'] = "cdfc53ab5d45aa0defbb8880184c67ce39745b42"

class ImageNetLowRes(Dataset):
    def __init__(self, root, split, transform=None):
        super().__init__()
        self.root = root
        self.split = split
        if transform is None:
            transform = lambda x: x
        self.transform = transform
        self._load()

    def _load(self):
        with open(Path(self.root) / 'map.txt', 'r') as f:
            remap = [int(x) for x in f.read().split('\n')]

        if self.split == 'train':
            Xs = []
            ys = []
            for i in range(1, 11):
                name = Path(self.root) / f"train_data_batch_{i}"
                with open(name, 'rb') as f:
                    d = pickle.load(f)
                    Xs.append(d['data'].reshape((-1, 3, 64, 64)).transpose((0, 2, 3, 1)))
                    ys.append(d['labels'])

            self.X = np.concatenate(Xs, axis=0)
            self.y = np.array([remap[x] for x in (np.concatenate(ys, axis=0) - 1)])
        elif self.split == 'val':
            name = Path(self.root) / f"val_data"

            with open(name, 'rb') as f:
                d = pickle.load(f)
            self.X = d['data'].reshape((-1, 3, 64, 64)).transpose((0, 2, 3, 1))
            self.y = np.array([remap[x - 1] for x in d['labels']])

    def __getitem__(self, index):
        return self.transform(self.X[index, ...]), self.y[index]

    def __len__(self):
        assert self.X.shape[0] == self.y.shape[0]
        return self.X.shape[0]


def main():
    wandb.init(project="topog", entity="pmin")

    wandb.config = {
        "version": "initial"
    }

    train_transform = transforms.Compose(
        [
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(.2, .2, .2, .2),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

    test_transform = transforms.Compose(
        [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])


    train_dataset = ImageNetLowRes('/home/pmin/hdd/imagenet64', 'train', transform=train_transform)
    test_dataset = ImageNetLowRes('/home/pmin/hdd/imagenet64', 'val', transform=test_transform)

    net = TopNet18()
    net = net.to(device='cuda')
    writer = SummaryWriter()

    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True, pin_memory=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=True, pin_memory=True, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=.3, patience=5)

    ema = ExponentialMovingAverage(net.parameters(), decay=0.995)

    epoch = 0
    i = 0
    running_loss = 0

    for epoch in tqdm(range(100)):
        j = 0
        for X, y in train_loader:
            net.train()
            i += X.shape[0]
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(X.to(device='cuda'))
            loss = criterion(outputs, y.to(device='cuda'))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss = loss.item()
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss:.3f}')
            writer.add_scalar('loss/train', i, running_loss)
            wandb.log({"train_loss": running_loss, "epoch": epoch, "batch": j})
            running_loss = 0.0
            j += 1
            ema.update()


        if epoch % 5 == 0:
            with ema.average_parameters():
                running_loss = 0
                total_examples = 0
                net.eval()
                for X, y in test_loader:
                    with torch.no_grad():
                        outputs = net(X.to(device='cuda'))
                        loss = criterion(outputs, y.to(device='cuda'))

                        # print statistics
                        running_loss += loss.item() * X.shape[0]

                    total_examples += X.shape[0]

            running_loss = running_loss / total_examples
            print(f'[{epoch + 1}] val loss: {running_loss:.5f}')
            writer.add_scalar('loss/val', i, running_loss)
            wandb.log({"val_loss": running_loss, "epoch": epoch})

            scheduler.step(running_loss)
                
        if epoch % 5 == 4:
            torch.save(net.state_dict(), f'/home/pmin/hdd/checkpoints/imagenet_ema_rn_trunc_epoch{epoch}.pt')

    wandb.finish()


if __name__ == '__main__':
    main()
