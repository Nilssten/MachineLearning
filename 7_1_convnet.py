from collections import Counter

import math
import torch
import numpy as np
import torchvision
import matplotlib
import matplotlib.pyplot as plt
from torch.hub import download_url_to_file
import os
import pickle
import torch.utils.data
import torch.nn.functional as F
from torch.utils.data import Subset
from tqdm import tqdm
import sklearn.model_selection

plt.rcParams["figure.figsize"] = (15, 5)
plt.style.use('dark_background')

LEARNING_RATE = 1e-4
BATCH_SIZE = 128
MAX_LEN = 200
TRAIN_TEST_SPLIT = 0.7
DEVICE = 'cpu'

if torch.cuda.is_available():
    DEVICE = 'cuda'
    MAX_LEN = 0


class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        path_dataset = '../data/Fruits28.pkl'
        if not os.path.exists(path_dataset):
            pass
            os.makedirs('../data', exist_ok=True)
            download_url_to_file(
                'http://share.yellowrobot.xyz/1645110979-deep-learning-intro-2022-q1/Fruits28.pkl',
                path_dataset,
                progress=True
            )
        with open(path_dataset, 'rb') as fp:
            X, Y, self.labels = pickle.load(fp)
        self.Y_idx = Y

        Y_counter = Counter(Y)
        Y_counts = np.array(list(Y_counter.values()))
        self.Y_weights = (1.0 / Y_counts) * np.sum(Y_counts)

        X = torch.from_numpy(np.array(X).astype(np.float32))
        self.X = X.permute(0, 3, 1, 2)
        self.input_size = self.X.size(-1)
        Y = torch.LongTensor(Y)
        self.Y = F.one_hot(Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]

        return x, y


dataset_full = Dataset()
train_test_split = int(len(dataset_full) * TRAIN_TEST_SPLIT)

idxes_train, idxes_test = sklearn.model_selection.train_test_split(
    np.arange(len(dataset_full)),
    train_size=train_test_split,
    test_size=len(dataset_full) - train_test_split,
    stratify=dataset_full.Y_idx,
    random_state=0
)

# For debugging
if MAX_LEN:
    idxes_train = idxes_train[:MAX_LEN]
    idxes_test = idxes_test[:MAX_LEN]

dataset_train = Subset(dataset_full, idxes_train)
dataset_test = Subset(dataset_full, idxes_test)

dataloader_train = torch.utils.data.DataLoader(
    dataset=dataset_train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=(len(dataset_train) % BATCH_SIZE == 1)
)

dataloader_test = torch.utils.data.DataLoader(
    dataset=dataset_test,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=(len(dataset_test) % BATCH_SIZE == 1)
)


def get_out_size(in_size, padding, kernel_size, stride):
    out = int((in_size + 2 * padding - kernel_size) / stride) + 1
    return out


class Conv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.K = torch.nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        torch.nn.init.kaiming_uniform_(self.K, a=math.sqrt(5))

    def forward(self, x):
        return F.conv2d(x, self.K, stride=self.stride, padding=self.padding)


class BatchNorm2d(torch.nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.gamma = torch.nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.beta = torch.nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.running_mean = torch.zeros(1, num_features, 1, 1, device=DEVICE)
        self.running_var = torch.ones(1, num_features, 1, 1, device=DEVICE)
        self.momentum = 0.1

    def forward(self, x):
        if self.training:
            mean = x.mean([0, 2, 3], keepdim=True)
            var = x.var([0, 2, 3], keepdim=True, unbiased=False)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
            x_normalized = (x - mean) / torch.sqrt(var + 1e-5)
        else:
            x_normalized = (x - self.running_mean) / torch.sqrt(self.running_var + 1e-5)
        return self.gamma * x_normalized + self.beta

    def to(self, device):
        super().to(device)
        self.running_mean = self.running_mean.to(device)
        self.running_var = self.running_var.to(device)
        return self




class MaxPool2d(torch.nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        batch_size = x.size(0)
        channels = x.size(1)
        in_size = x.size(-1)  # last dim from (B, C, W, H)
        out_size = get_out_size(in_size, self.padding, self.kernel_size, self.stride)

        out = x  # TODO

        return out


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            Conv2d(in_channels=3, out_channels=5, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(num_features=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            Conv2d(in_channels=5, out_channels=9, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(num_features=9),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=4, stride=2, padding=1),
            Conv2d(in_channels=9, out_channels=9, kernel_size=3, stride=1, padding=1)
        )

        final_size = 28
        final_size = get_out_size(final_size, 1, 3, 1)  # First conv layer
        final_size = get_out_size(final_size, 1, 4, 2)  # First maxpool
        final_size = get_out_size(final_size, 1, 3, 1)  # Second conv layer
        final_size = get_out_size(final_size, 1, 4, 2)  # Second maxpool
        final_size = get_out_size(final_size, 1, 3, 1)  # Third conv layer

        self.fc = torch.nn.Linear(
            in_features=final_size * final_size * 9,
            out_features=67
        )

    def forward(self, x):
        features = self.encoder(x)
        flatten = features.view(features.size(0), -1)
        logits = self.fc(flatten)
        y_prim = torch.softmax(logits, dim=1)
        return y_prim


model = Model()
model = model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

metrics = {}
for stage in ['train', 'test']:
    for metric in [
        'loss',
        'acc'
    ]:
        metrics[f'{stage}_{metric}'] = []

for epoch in range(1, 100):
    for data_loader in [dataloader_train, dataloader_test]:
        metrics_epoch = {key: [] for key in metrics.keys()}

        if data_loader == dataloader_test:
            stage = 'test'
            model = model.eval()
            torch.set_grad_enabled(False)
        else:
            stage = 'train'
            model = model.train()
            torch.set_grad_enabled(True)

        for x, y in tqdm(data_loader):

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            y_prim = model.forward(x)

            y_idx = y.cpu().data.numpy().argmax(axis=-1)
            w = torch.FloatTensor(dataset_full.Y_weights[y_idx]).unsqueeze(dim=-1).to(DEVICE)

            loss = -torch.mean(y * w * torch.log(y_prim + 1e-8))

            if data_loader == dataloader_train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            np_y_prim = y_prim.cpu().data.numpy()
            np_y = y.cpu().data.numpy()
            x = x.cpu()

            idx_y = np.argmax(np_y, axis=1)
            idx_y_prim = np.argmax(np_y_prim, axis=1)

            acc = np.average((idx_y == idx_y_prim) * 1.0)

            metrics_epoch[f'{stage}_acc'].append(acc)
            metrics_epoch[f'{stage}_loss'].append(loss.cpu().item())

        metrics_strs = []
        for key in metrics_epoch.keys():
            if stage in key:
                value = np.mean(metrics_epoch[key])
                metrics[key].append(value)
                metrics_strs.append(f'{key}: {round(value, 2)}')

        print(f'epoch: {epoch} {" ".join(metrics_strs)}')

    plt.clf()
    plt.subplot(121)  # row col idx
    plts = []
    c = 0
    for key, value in metrics.items():
        plts += plt.plot(value, f'C{c}', label=key)
        ax = plt.twinx()
        c += 1

    plt.legend(plts, [it.get_label() for it in plts])

    for i, j in enumerate([4, 5, 6, 10, 11, 12, 16, 17, 18]):
        plt.subplot(3, 6, j)
        color = 'green' if idx_y[i] == idx_y_prim[i] else 'red'
        plt.title(f"pred: {dataset_full.labels[idx_y_prim[i]]}\n real: {dataset_full.labels[idx_y[i]]}", color=color)
        plt.imshow(x[i].permute(1, 2, 0))

    plt.tight_layout(pad=0.5)
    plt.show()
