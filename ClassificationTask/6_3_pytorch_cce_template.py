import os
import pickle
from collections import Counter
import torch
import numpy as np
from torch.hub import download_url_to_file
import matplotlib.pyplot as plt
import torch.utils.data
import torch.nn.functional as F
import sklearn.model_selection
from torch.utils.data import Subset

# Setting up plot parameters
plt.rcParams["figure.figsize"] = (12, 12)
plt.style.use('dark_background')

# Constants
LEARNING_RATE = 1e-3
BATCH_SIZE = 16
TRAIN_TEST_SPLIT = 0.7
EMBEDDING_SIZE = 8

class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        path_dataset = '../data/cardekho_india_dataset_cce.pkl'
        if not os.path.exists(path_dataset):
            os.makedirs('../data', exist_ok=True)
            download_url_to_file(
                'http://share.yellowrobot.xyz/1645110979-deep-learning-intro-2022-q1/cardekho_india_dataset_cce.pkl',
                path_dataset,
                progress=True
            )
        with open(f'{path_dataset}', 'rb') as fp:
            X, Y, self.labels = pickle.load(fp)

        # Convert to numpy array
        X = np.array(X)
        self.Y_idx = Y
        self.Y_labels = self.labels[3]
        self.Y_len = len(self.Y_labels)

        # Calculate class weights
        class_counts = Counter(self.Y_idx)
        total_samples = sum(class_counts.values())
        class_weights = {cls: total_samples / count for cls, count in class_counts.items()}

        # Convert class weights to a tensor
        self.Y_weights = torch.tensor([class_weights[cls] for cls in range(len(self.labels[3]))], dtype=torch.float32)

        # Features and labels
        self.X_classes = np.array(X[:, :3])
        self.X = np.array(X[:, 3:], dtype=np.float32)  # Convert to float32
        X_mean = np.mean(self.X, axis=0)
        X_std = np.std(self.X, axis=0)
        self.X = (self.X - X_mean) / X_std  # Standardize

        # One-hot encode self.Y
        self.Y = torch.nn.functional.one_hot(torch.tensor(self.Y_idx), num_classes=self.Y_len).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], np.array(self.X_classes[idx]), self.Y_idx[idx]

# Splitting dataset into train and test
dataset_full = Dataset()
train_test_split = int(len(dataset_full) * TRAIN_TEST_SPLIT)
idxes_train, idxes_test = sklearn.model_selection.train_test_split(
    np.arange(len(dataset_full)),
    train_size=train_test_split,
    test_size=len(dataset_full) - train_test_split,
    stratify=dataset_full.Y_idx,
    random_state=0
)

# Creating data loaders
dataloader_train = torch.utils.data.DataLoader(
    dataset=Subset(dataset_full, idxes_train),
    batch_size=BATCH_SIZE,
    shuffle=True
)
dataloader_test = torch.utils.data.DataLoader(
    dataset=Subset(dataset_full, idxes_test),
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Model definition
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.embs = torch.nn.ModuleList()
        for i in range(3):
            self.embs.append(
                torch.nn.Embedding(
                    embedding_dim=EMBEDDING_SIZE,
                    num_embeddings=len(dataset_full.labels[i])
                )
            )

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=3 * EMBEDDING_SIZE + 4, out_features=20),
            torch.nn.LogSoftmax(dim=1),
            torch.nn.Linear(in_features=20, out_features=10),
            torch.nn.LogSoftmax(dim=1),
            torch.nn.Linear(in_features=10, out_features=4)
        )

    def forward(self, x, x_classes):
        x_emb = torch.cat([self.embs[i](x_classes[:, i]) for i in range(3)], dim=-1)
        x = torch.cat([x, x_emb], dim=-1)
        return self.layers(x)

# Custom loss function
class LossCCE(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_prim, y):
        y_prim_softmax = F.softmax(y_prim, dim=1)
        predicted_probabilities = y_prim_softmax[range(len(y)), y]
        negative_log_probabilities = -torch.log(predicted_probabilities)
        return torch.mean(negative_log_probabilities)

def calculate_accuracy(y_pred, y_true):
    _, predicted = torch.max(y_pred, 1)
    correct = (predicted == y_true).sum().item()
    return correct / len(y_true)

def calculate_f1_score(conf_matrix):
    tp = np.diag(conf_matrix)
    fp = np.sum(conf_matrix, axis=0) - tp
    fn = np.sum(conf_matrix, axis=1) - tp
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    return np.nanmean(f1)

# Model, optimizer, loss function
model = Model()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
loss_fn = LossCCE()

# Lists to store metrics
loss_plot_train = []
loss_plot_test = []
acc_plot_train = []
acc_plot_test = []
f1_plot_train = []
f1_plot_test = []
conf_matrix_train = np.zeros((dataset_full.Y_len, dataset_full.Y_len))
conf_matrix_test = np.zeros((dataset_full.Y_len, dataset_full.Y_len))

# Training loop
for epoch in range(1, 1000):

    for dataloader in [dataloader_train, dataloader_test]:

        model.train() if dataloader == dataloader_train else model.eval()
        torch.set_grad_enabled(dataloader == dataloader_train)

        losses = []
        accs = []
        f1s = []

        conf_matrix = np.zeros((dataset_full.Y_len, dataset_full.Y_len))

        for x, x_classes, y in dataloader:

            # Forward pass
            y_prim = model(x, x_classes)

            # Compute loss
            loss = loss_fn(y_prim, y)
            losses.append(loss.item())

            # Compute accuracy
            acc = calculate_accuracy(y_prim, y)
            accs.append(acc)

            # Update confusion matrix
            for true, pred in zip(y, torch.argmax(y_prim, dim=1)):
                conf_matrix[true.item(), pred.item()] += 1

            # Backward pass and optimization (only in training mode)
            if dataloader == dataloader_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Compute F1-score
        f1 = calculate_f1_score(conf_matrix)
        f1s.append(f1)

        # Store metrics
        if dataloader == dataloader_train:
            loss_plot_train.append(np.mean(losses))
            acc_plot_train.append(np.mean(accs))
            conf_matrix_train = conf_matrix
            f1_plot_train.append(np.mean(f1s))
        else:
            loss_plot_test.append(np.mean(losses))
            acc_plot_test.append(np.mean(accs))
            conf_matrix_test = conf_matrix
            f1_plot_test.append(np.mean(f1s))

    # Print metrics
    print(
        f'epoch: {epoch} '
        f'loss_train: {loss_plot_train[-1]} '
        f'loss_test: {loss_plot_test[-1]} '
        f'acc_train: {acc_plot_train[-1]} '
        f'acc_test: {acc_plot_test[-1]} '
        f'f1_train: {f1_plot_train[-1]} '
        f'f1_test: {f1_plot_test[-1]} '
    )

    # Plotting
    if epoch % 10 == 0:
        plt.tight_layout(pad=0)
        fig, axes = plt.subplots(nrows=2, ncols=2)

        # Loss plot
        ax1 = axes[0, 0]
        ax1.plot(loss_plot_train, 'r-', label='train')
        ax2 = ax1.twinx()
        ax2.plot(loss_plot_test, 'c-', label='test')
        ax1.legend()
        ax2.legend(loc='upper left')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")

        # Accuracy plot
        ax1 = axes[0, 1]
        ax1.plot(acc_plot_train, 'r-', label='train')
        ax2 = ax1.twinx()
        ax2.plot(acc_plot_test, 'c-', label='test')
        ax1.legend()
        ax2.legend(loc='upper left')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")

        # Confusion matrix
        for ax, conf_matrix in [(axes[1, 0], conf_matrix_train), (axes[1, 1], conf_matrix_test)]:
            ax.imshow(conf_matrix, interpolation='nearest', cmap=plt.get_cmap('Greys'))
            ax.set_xticks(np.arange(dataset_full.Y_len))
            ax.set_xticklabels(dataset_full.Y_labels, rotation=45)
            ax.set_yticks(np.arange(dataset_full.Y_len))
            ax.set_yticklabels(dataset_full.Y_labels)
            for x in range(dataset_full.Y_len):
                for y in range(dataset_full.Y_len):
                    perc = round(100 * conf_matrix[x, y] / np.sum(conf_matrix))
                    ax.annotate(
                        str(int(conf_matrix[x, y])),
                        xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        backgroundcolor=(1., 1., 1., 0.),
                        color='green' if perc < 50 else 'white',
                        fontsize=10
                    )
            ax.set_xlabel('True')
            ax.set_ylabel('Predicted')

        plt.show()