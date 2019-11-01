"""
Mô hình này sử dụng 2 mạng LSTM
1 mạng sử dụng ở level word embedding, mạng còn lại ở level char embedding
Dataset sử dụng nltk
"""

import nltk
import pickle as pkl

"""
'''Download dataset'''
nltk.download('treebank')
nltk.download('universal_tagset')
"""

tagged_sentence = nltk.corpus.treebank.tagged_sents(tagset='universal')

print("Number of tagged sentences: ", len(tagged_sentence))

print(tagged_sentence[0])

"""
tagged_sentence la mot mang cac sentence, moi sentence la list of tuple. Here a sentence in tagged_sentence
[('Pierre', 'NOUN'), ('Vinken', 'NOUN'), (',', '.'), ('61', 'NUM'), ('years', 'NOUN'), ('old', 'ADJ'), (',', '.'), ('will', 'VERB'), ('join', 'VERB'), ('the', 'DET'), ('board', 'NOUN'), ('as', 'ADP'), ('a', 'DET'), ('nonexecutive', 'ADJ'), ('director', 'NOUN'), ('Nov.', 'NOUN'), ('29', 'NUM'), ('.', '.')]
"""

"""
Xay dung cac dictionary sau:
word2idx: tu dien anh xa moi tu sang 1 unique integer
tag2idx: tu dien anh xa moi tag sang 1 unique integer
char2idx: tu dien anh xa moi character sang 1 unique integer
"""

"""Do da dump cac dictionary nen se comment doan build dictionary lai"""

"""
word2idx = {}
tag2idx = {}
char2idx = {}

for sentence in tagged_sentence:
    for word, tag in sentence:
        if word not in word2idx.keys():
            word2idx[word] = len(word2idx)
        if tag not in tag2idx.keys():
            tag2idx[tag] = len(tag2idx)
        for char in word:
            if char not in char2idx.keys():
                char2idx[char] = len(char2idx)

# dump word2idx, tag2idx, char2idx
with open('./dataset/word2idx.pkl', 'wb') as f:
    pkl.dump(word2idx, f)
with open('./dataset/tag2idx.pkl', 'wb') as f:
    pkl.dump(tag2idx, f)
with open('./dataset/char2idx.pkl', 'wb') as f:
    pkl.dump(char2idx, f)

"""

# load word2idx, tag2idx, char2idx
with open('./dataset/word2idx.pkl', 'rb') as f:
    word2idx = pkl.load(f)
with open('./dataset/tag2idx.pkl', 'rb') as f:
    tag2idx = pkl.load(f)
with open('./dataset/char2idx.pkl', 'rb') as f:
    char2idx = pkl.load(f)

print("Number of unique words: ", len(word2idx))
print("Number of unique tags: ", len(tag2idx))
print("NUmber of unique chars: ", len(char2idx))
word_vocab_size = len(word2idx)
char_vocab_size = len(char2idx)
tag_vocab_size = len(tag2idx)

"""SPlit dataset to train and test"""

import random

tr_random = random.sample(list(range(len(tagged_sentence))), int(0.95 * len(tagged_sentence)))

train = [tagged_sentence[i] for i in tr_random]
test = [tagged_sentence[i] for i in range(len(tagged_sentence)) if i not in tr_random]

print(test)

"""Define hyper parameter"""
WORD_EMBEDDING_DIM = 1024
CHAR_EMBEDDING_DIM = 128
WORD_HIDDEN_DIM = 1024
CHAR_HIDDEN_DIM = 1024
EPOCHS = 70

import torch
import torch.nn as nn
from torch.functional import F
import torch.optim as optim


class DualLSTMTagger(nn.Module):
    def __init__(self, word_embedding_dim, word_hidden_dim,
                 char_embedding_dim, char_hidden_dim,
                 word_vocab_size, char_vocab_size, tag_vocab_size):
        super(DualLSTMTagger, self).__init__()

        self.char_embedding = nn.Embedding(num_embeddings=char_vocab_size, embedding_dim=char_embedding_dim)
        self.word_embedding = nn.Embedding(num_embeddings=word_vocab_size, embedding_dim=word_embedding_dim)
        self.char_lstm = nn.LSTM(input_size=char_embedding_dim, hidden_size=char_hidden_dim)
        self.lstm = nn.LSTM(input_size=(word_embedding_dim + char_hidden_dim), hidden_size=word_hidden_dim)

        self.hidden2tag = nn.Linear(in_features=word_hidden_dim, out_features=tag_vocab_size)

    def foward(self, sent, words):
        embeds = self.word_embedding(sent)

        char_hidden_final = []

        for word in words:
            char_embeds = self.char_embedding(word)
            char_out, (char_hn, char_cn) = self.char_lstm(char_embeds.view(len(word), 1, -1))
            char_hidden_state_of_word = char_hn.view(-1)
            char_hidden_final.append(char_hidden_state_of_word)
        char_hidden_final = torch.stack(tuple(char_hidden_final))
        combined = torch.cat((embeds, char_hidden_final), 1)
        lstm_out, (hn, cn) = self.lstm(combined.view(len(sent), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sent), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


model = DualLSTMTagger(WORD_EMBEDDING_DIM, WORD_HIDDEN_DIM, CHAR_EMBEDDING_DIM, CHAR_HIDDEN_DIM,
                       word_vocab_size, char_vocab_size, tag_vocab_size)
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

loss_function = nn.NLLLoss()

optimizer = optim.Adam(model.parameters(), lr=0.01)


def sequence_to_idx(sequence, ix):
    return torch.tensor([ix[s] for s in sequence], dtype=torch.long)


# The test sentence
seq = "everybody eat the food . I kept looking out the window , trying to find the one I was waiting for .".split()
print("Running a check on the model before training.\nSentences:\n{}".format(" ".join(seq)))
words = [torch.tensor(sequence_to_idx(s[0], char2idx), dtype=torch.long).to(device) for s in seq]
for s in seq:
    print(s[0])
print("words: ", words)
print(len(words))
sentence = torch.tensor(sequence_to_idx(seq, word2idx), dtype=torch.long).to(device)
print("sent: ", sentence)

# with torch.no_grad():
#     words = [torch.tensor(sequence_to_idx(s[0], char2idx), dtype=torch.long).to(device) for s in seq]
#     sentence = torch.tensor(sequence_to_idx(seq, word2idx), dtype=torch.long).to(device)
#
#     tag_scores = model(sentence, words)
#     _, indices = torch.max(tag_scores, 1)
#     ret = []
#     for i in range(len(indices)):
#         for key, value in tag2idx.items():
#             if indices[i] == value:
#                 ret.append((seq[i], key))
#     print(ret)
# Training start
