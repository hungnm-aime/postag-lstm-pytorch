import torch
from collections import Counter
import torch.nn as nn
from torch.functional import F
import torch.optim as optim


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]

    return torch.tensor(idxs, dtype=torch.long)


training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]

word2index = {}
vocab = Counter()
for sent, tags in training_data:
    vocab.update(sent)
print(vocab)
word2index = {w: i for i, w in enumerate(vocab)}
print(word2index)

tag_to_ix = {"DET": 0, "NN": 1, "V": 2}

EMBEDDING_DIM = 6
HIDDEN_DIM = 6


class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()

        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim)

        self.hidden2tag = nn.Linear(in_features=hidden_dim, out_features=tagset_size)

    def forward(self, sent):
        embeds = self.word_embeddings(sent)
        '''input cua lstm se co dang [seqlen, batch, input].'''
        lstm_out, (hn, cn) = self.lstm(embeds.view(len(sent), 1, -1))  # [seqlen, 1, embedding_dim]
        tag_space = self.hidden2tag(lstm_out.view(len(sent), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word2index), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word2index)
    tag_scores = model(inputs)
    print(tag_scores)

for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        sentence_in = prepare_sequence(sentence, word2index)
        targets = prepare_sequence(tags, tag_to_ix)

        # Step 3. Run our forward pass.
        tag_scores = model(sentence_in)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

print("AFTER TRAIN")
# See what the scores are after training
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word2index)
    tag_scores = model(inputs)

    # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
    # for word i. The predicted tag is the maximum scoring tag.
    # Here, we can see the predicted sequence below is 0 1 2 0 1
    # since 0 is index of the maximum value of row 1,
    # 1 is the index of maximum value of row 2, etc.
    # Which is DET NOUN VERB DET NOUN, the correct sequence!
    print(tag_scores)
