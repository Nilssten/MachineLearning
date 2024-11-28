import json
import os
import pdb
import pickle

import imageio #pip3 install imageio
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence, pack_padded_sequence
from torch.hub import download_url_to_file
import torch.utils.data

import argparse
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-run_name', default='', type=str)
parser.add_argument('-batch_size', default=128, type=int)
parser.add_argument('-epochs', default=1000, type=int)
parser.add_argument('-learning_rate', default=1e-3, type=float)

parser.add_argument('-hidden_size', default=64, type=int)
parser.add_argument('-max_len', default=600, type=int)

args, args_other = parser.parse_known_args()

BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
LEARNING_RATE = args.learning_rate

HIDDEN_SIZE = args.hidden_size
TRANSFORMER_LAYERS = 8
DROPOUT = 0.1
run_name = args.run_name

TRANSFORMER_HEADS = 4

DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'

MIN_SENTENCE_LEN = 3
MAX_SENTENCE_LEN = 20
MAX_LEN = args.max_len

PATH_DATA = '../data'
os.makedirs('./results', exist_ok=True)
os.makedirs(PATH_DATA, exist_ok=True)

class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        path_dataset = '../data/quotes.pkl'
        if not os.path.exists(path_dataset):
            os.makedirs('../data', exist_ok=True)
            download_url_to_file(
                'http://share.yellowrobot.xyz/1645110979-deep-learning-intro-2022-q1/quotes.pkl',
                path_dataset,
                progress=True
            )

        with open(path_dataset, 'rb') as fp:
            (
                self.final_quotes_sentences, self.final_authors, self.final_categories,
                self.vocabulary_keys, self.vocabulary_counts, self.authors_keys, self.categories_keys
            ) = pickle.load(fp)
        self.max_sentence_length = np.max([len(it) for it in self.final_quotes_sentences])

    def __len__(self):
        if MAX_LEN:
            return MAX_LEN
        return len(self.final_quotes_sentences)

    def __getitem__(self, idx):
        x_raw = np.array(self.final_quotes_sentences[idx], dtype=np.int64)

        y = np.roll(x_raw, -1) # move all words by one position left
        y = y[:-1]
        x = x_raw[:-1]
        x_length = len(x)

        pad_right = self.max_sentence_length - x_length
        pad_left = 0
        x_padded = np.pad(x, (pad_left, pad_right))
        y_padded = np.pad(y, (pad_left, pad_right))

        return x_padded, y_padded, x_length


dataset_full = Dataset()
train_test_split = int(len(dataset_full) * 0.8)
dataset_train, dataset_test = torch.utils.data.random_split(
    dataset_full,
    [train_test_split, len(dataset_full) - train_test_split],
    generator=torch.Generator().manual_seed(0)
)

data_loader_train = torch.utils.data.DataLoader(
    dataset=dataset_train,
    batch_size=BATCH_SIZE,
    shuffle=True
)
data_loader_test = torch.utils.data.DataLoader(
    dataset=dataset_test,
    batch_size=BATCH_SIZE,
    shuffle=False
)


class PositionalEncoding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()

        mat = torch.zeros(num_embeddings, embedding_dim).to(DEVICE)
        pos = torch.arange(0, num_embeddings, dtype=torch.float)
        pos = pos.unsqueeze(dim=1)

        half = torch.arange(0, embedding_dim, step=2, dtype=torch.float)
        div = torch.exp(half * (-np.log(10000) / embedding_dim))
        #div = half / (10000 ** embedding_dim)

        mat[:, 0::2] = torch.sin(pos * div)
        mat[:, 1::2] = torch.cos(pos * div)

        self.pe = mat

    def forward(self, idxes):
        return self.pe[idxes, :]


class TransformerLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.ff_k = torch.nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE)
        self.ff_q = torch.nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE)
        self.ff_v = torch.nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE)

        self.ff_mid = torch.nn.Sequential(
            torch.nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE)
        )

        self.norm_1 = torch.nn.LayerNorm(normalized_shape=HIDDEN_SIZE)
        self.norm_2 = torch.nn.LayerNorm(normalized_shape=HIDDEN_SIZE)

    def forward(self, x, lengths, atten):
        batch_size = x.size(0) # x.shape (B, seq, HIDDEN_SIZE)
        seq_size = x.size(1)
        dim_size = x.size(2)

        k = self.ff_k.forward(x)
        q = self.ff_q.forward(x)
        v = self.ff_v.forward(x)

        atten_law = (q @ k.transpose(-1, -2)) / np.sqrt(dim_size)

        mask = torch.tril(torch.ones(seq_size, seq_size)).to(DEVICE)
        atten_raw = atten_law.masked_fill(mask == 0, value=float("-inf"))

        for idx, length in enumerate(lengths):
            atten_raw[idx, :, length:] = float("-inf")
            atten_raw[idx, length:, :] = float("-inf")

        atten = torch.softmax(atten_raw, dim=1)
        out = atten @ v

        out_1 = x + torch.dropout(out, p=DROPOUT, train=self.training)
        out_1_norm = self.norm_1.forward(out_1)

        out_2 = self.ff_mid.forward(out_1_norm)
        out_3 = out_2 + out_1_norm #skip connection
        y_prim = self.norm_2.forward(out_3)

        return y_prim, lengths, atten


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.emb_w = torch.nn.Embedding(
            num_embeddings=len(dataset_full.vocabulary_keys),
            embedding_dim=HIDDEN_SIZE
        )

        self.emb_p = PositionalEncoding(
            num_embeddings=len(dataset_full.vocabulary_keys),
            embedding_dim=HIDDEN_SIZE
        )

        self.layers = torch.nn.ModuleList(
            [TransformerLayer() for _ in range(TRANSFORMER_LAYERS)]
        )

        self.fc = torch.nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE)

    def forward(self, x: PackedSequence):

        w_e = self.emb_w.forward(x.data)
        w_e = PackedSequence(
            data=w_e,
            batch_sizes=x.batch_sizes,
            sorted_indices=x.sorted_indices
        )

        w_e_unpacked, lenghts = pad_packed_sequence(w_e, batch_first=True)
        p_idx = torch.arange(0, int(torch.max(lenghts))).to(DEVICE)

        p_e = self.emb_p.forward(p_idx)
        p_e = p_e[None, :, :]
        p_e = p_e.expand(w_e_unpacked.size())

        z = w_e_unpacked + p_e
        for layer in self.layers:
            z, _, _ = layer.forward(z, lenghts, None)

        x_packed = pack_padded_sequence(z, lenghts, batch_first=True, enforce_sorted=False)
        out_fc = self.fc.forward(x_packed.data)

        y_prim_log = self.emb_w.weight @ out_fc[:, :, None]
        y_prim = torch.softmax(y_prim_log[:, :, 0], dim=-1)

        y_prim_packed = PackedSequence(
            data=y_prim,
            batch_sizes=x.batch_sizes,
            sorted_indices=x.sorted_indices
        )

        return y_prim_packed, None

model = Model()
model = model.to(DEVICE)

# W = W - dW * alpha
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

metrics = {}
best_test_loss = float('Inf')
for stage in ['train', 'test']:
    for metric in [
        'loss',
        'acc'
    ]:
        metrics[f'{stage}_{metric}'] = []

loss_weights = torch.FloatTensor(1 - dataset_full.vocabulary_counts / np.sum(dataset_full.vocabulary_counts))
loss_weights = loss_weights.to(DEVICE)

for epoch in range(1, EPOCHS+1):

    for data_loader in [data_loader_train, data_loader_test]:
        metrics_epoch = {key: [] for key in metrics.keys()}

        stage = 'train'
        if data_loader == data_loader_test:
            stage = 'test'

        for x, y, length in tqdm(data_loader):

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            x_packed = pack_padded_sequence(x, length, batch_first=True, enforce_sorted=False)
            y_packed = pack_padded_sequence(y, length, batch_first=True, enforce_sorted=False)

            y_prim_packed, atten = model.forward(x_packed)

            idxes_batch = range(len(y_packed.data))
            idxes_y = y_packed.data
            # CCE
            loss = -torch.mean(
                loss_weights[idxes_y] * torch.log(y_prim_packed.data[idxes_batch, idxes_y] + 1e-8)
            )

            idxes_y_prim = y_prim_packed.data.argmax(dim=-1)
            acc = torch.mean((idxes_y_prim == idxes_y) * 1.0)
            metrics_epoch[f'{stage}_acc'].append(acc.item())

            metrics_epoch[f'{stage}_loss'].append(loss.item()) # Tensor(0.1) => 0.1f

            if data_loader == data_loader_train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()


        metrics_strs = []
        for key in metrics_epoch.keys():
            if stage in key:
                value = np.mean(metrics_epoch[key])
                metrics[key].append(value)
                metrics_strs.append(f'{key}: {round(value, 4)}')
        print(f'epoch: {epoch} {" ".join(metrics_strs)}')

    if best_test_loss > loss.item():
        best_test_loss = loss.item()
        torch.save(model.cpu().state_dict(), f'./results/{run_name}-model-{epoch}.pt')
        model = model.to(DEVICE)

        # pip install imageio
        imageio.imwrite(f'./results/{run_name}-epoch-{epoch}-atten-0.png', atten[0].cpu().data.numpy())
        imageio.imwrite(f'./results/{run_name}-epoch-{epoch}-atten-l.png', atten[-1].cpu().data.numpy())


    print('Examples:')
    y_prim_unpacked, lengths_unpacked = pad_packed_sequence(y_prim_packed.cpu(), batch_first=True)
    y_prim_unpacked = y_prim_unpacked[:5] # 5 examples
    for idx, each in enumerate(y_prim_unpacked):
        length = lengths_unpacked[idx]

        y_prim_idxes = np.argmax(each[:length].data.numpy(), axis=1).tolist()
        x_idxes = x[idx, :length].cpu().data.numpy().tolist()
        y_prim_idxes = [x_idxes[0]] + y_prim_idxes
        print('x     : ' +' '.join([dataset_full.vocabulary_keys[it] for it in x_idxes]))
        print('y_prim: ' +' '.join([dataset_full.vocabulary_keys[it] for it in y_prim_idxes]))
        print('')

    plt.figure(figsize=(12,5))
    plts = []
    c = 0
    for key, value in metrics.items():
        plts += plt.plot(value, f'C{c}', label=key)
        ax = plt.twinx()
        c += 1

    plt.legend(plts, [it.get_label() for it in plts])
    plt.savefig(f'./results/{run_name}-epoch-{epoch}.png')
    plt.show()