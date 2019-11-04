from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from torch.functional import F
import torch.optim as optim

"""small dataset"""
training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("A cat eats the orange".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"]),
    ("Someone play the guitar".split(), ["NN", "V", "DET", "NN"])
]

word2index = {}
vocab = Counter()
for sent, tags in training_data:
    vocab.update(sent)
word2index = {w: i + 2 for i, w in enumerate(vocab)}
word2index['<PADDING>'] = 0
word2index['<OOV>'] = 1

tag2idx = {"<PAD>": 0, "DET": 1, "NN": 2, "V": 3}

EMBEDDING_DIM = 6
HIDDEN_DIM = 8
BATCH_SIZE = 2
MAX_LEN = 3


def sequence2idx(sequence, ix):
    """
    thực hiện convert data dạng chuỗi thành dạng index
    :param sequence:
    :param ix: dictionary word to index hoặc tag to index
    :return:
    """
    return [ix[s] for s in sequence]


def padding_truncating(x, y, max_len):
    """
    vì sử dụng batch_size nên độ dài của sentence phải cố định
    :param x:
    :param y:
    :param max_len:
    :return:
    """
    if len(x) > max_len:
        x = x[: max_len]
        y = y[: max_len]
    elif len(x) < max_len:
        pad = list(np.zeros((max_len - len(x)), dtype=int))
        x = x + pad
        y = y + pad
    return x, y


x_train = []
y_train = []
for data in training_data:
    x, y = sequence2idx(data[0], word2index), sequence2idx(data[1], tag2idx)
    x, y = padding_truncating(x, y, MAX_LEN)
    x_train.append(x)
    y_train.append(y)
x_train = np.array(x_train)
y_train = np.array(y_train)
tensor_dataset_train = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
train_loader = DataLoader(tensor_dataset_train, shuffle=True, batch_size=BATCH_SIZE)
print(train_loader)
for train in train_loader:
    print(train)
"""Xây dựng model"""


class LSTMTager(nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_size, output_size, batch_size, is_bidirection=False,
                 num_layers=1):
        super(LSTMTager, self).__init__()
        self.batch_size = batch_size
        self.is_bidirection = is_bidirection
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.word_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size)
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_size, output_size)

    def forward(self, sentence):
        embeds = self.word_embedding(sentence)
        # print("embeds: ", embeds)
        # print("size of embeds: ", embeds.size())
        lstm_out, (hn, cn) = self.lstm(embeds, (self.init_hidden()))
        # print("lstm_out: ", lstm_out)
        # print("size of lstm_out: ", lstm_out.size())
        tag_space = self.hidden2tag(lstm_out)
        # print("tag_space: ", tag_space)
        # print("size of tag_space: ", tag_space.size())
        tag_scores = F.log_softmax(tag_space, dim=1)
        # print("tag_scores: ", tag_scores)
        # print("size of tag_score: ", tag_scores.size())

        return tag_scores

    def init_hidden(self):
        if self.is_bidirection:
            print("bi direction...")
            h0 = torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_size)
            c0 = torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_size)
        else:
            print("one direction...")
            h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
            c0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        return h0, c0


model = LSTMTager(len(word2index), EMBEDDING_DIM, HIDDEN_DIM, len(tag2idx), BATCH_SIZE)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
with torch.no_grad():
    for train_x, train_y in train_loader:
        tag_scores = model(train_x)
        print(tag_scores)

epochs = 1
for epoch in range(epochs):

    for train_x, train_y in train_loader:
        model.zero_grad()

        tag_scores = model(train_x)

        print("tag_scores: ", tag_scores)

        _, indices = torch.max(tag_scores, 1)
        print("_: ", _)
        print("indices: ", indices)

        print("train_y: ", train_y)
        print("size of train_y: ", train_y.size())
        loss = loss_function(tag_scores, train_y)
        loss.backward()
        optimizer.step()