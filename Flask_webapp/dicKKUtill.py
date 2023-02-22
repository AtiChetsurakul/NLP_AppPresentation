from torchtext.data.utils import get_tokenizer
import torch
import torch.nn as nn
from spacy.lang.en.stop_words import STOP_WORDS
import spacy
import pickle

import torch.nn as nn
import torch


class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers, dropout_rate):

        super().__init__()
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, num_layers=num_layers,
                            dropout=dropout_rate, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        # when you do LM, you look forward, so it does not make sense to do bidirectional
        self.fc = nn.Linear(hid_dim, vocab_size)

    def init_hidden(self, batch_size, device):
        # this function gonna be run in the beginning of the epoch
        hidden = torch.zeros(self.num_layers, batch_size,
                             self.hid_dim).to(device)
        cell = torch.zeros(self.num_layers, batch_size,
                           self.hid_dim).to(device)

        return hidden, cell  # return as tuple

    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden = hidden.detach()  # removing this hidden from gradients graph
        cell = cell.detach()  # removing this hidden from gradients graph
        return hidden, cell

    def forward(self, src, hidden):
        # src: [batch_size, seq_len]

        # embed
        embedded = self.embedding(src)
        # embed : [batch_size, seq_len, emb_dim]

        # send this to the lstm
        # we want to put hidden here... because we want to reset hidden .....
        output, hidden = self.lstm(embedded, hidden)
        # output : [batch_size, seq_len, hid_dim] ==> all hidden states
        # hidden : [batch_size, seq_len, hid_dim] ==> last hidden states from each layer

        output = self.dropout(output)
        prediction = self.fc(output)
        # prediction: [batch size, seq_len, vocab_size]
        return prediction, hidden


def generate(prompt, max_seq_len, temperature, model, tokenizer, vocab, device, seed=3407):
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    tokens = tokenizer(prompt)
    indices = [vocab[t] for t in tokens]
    batch_size = 1
    hidden = model.init_hidden(batch_size, device)
    with torch.no_grad():
        for i in range(max_seq_len):
            src = torch.LongTensor([indices]).to(device)
            prediction, hidden = model(src, hidden)

            # prediction: [batch size, seq len, vocab size]
            # prediction[:, -1]: [batch size, vocab size] #probability of last vocab

            probs = torch.softmax(prediction[:, -1] / temperature, dim=-1)
            prediction = torch.multinomial(probs, num_samples=1).item()

            # if it is unk, we sample again
            while prediction == vocab['<unk>']:
                prediction = torch.multinomial(probs, num_samples=1).item()

            if prediction == vocab['<eos>']:  # if it is eos, we stop
                break

            # autoregressive, thus output becomes input
            indices.append(prediction)

    itos = vocab.get_itos()
    tokens = [itos[i] for i in indices]
    return tokens


tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
nlp = spacy.load('en_core_web_md')
with open('predickapp/vocab.pickle', 'rb') as handle:
    vocab = pickle.load(handle)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vocab_size = len(vocab)
emb_dim = 400                # 400 in the paper
hid_dim = 1150               # 1150 in the paper
num_layers = 3                # 3 in the paper
dropout_rate = 0.5
lr = 1e-3

model = LSTMLanguageModel(vocab_size, emb_dim, hid_dim,
                          num_layers, dropout_rate).to(device)

model.load_state_dict(torch.load('predickapp/predictor_weight.pt'))

# prompt = 'for i in'
# max_seq_len = 30
# seed = 3407

# generation = generate(prompt, max_seq_len, .8, model, tokenizer,
#                       vocab, device, seed)
# print(str(.8)+'     '+' '.join(generation))

# dickle = (generate, LSTMLanguageModel(vocab_size, emb_dim, hid_dim,
#                                       num_layers, dropout_rate), tokenizer, nlp, device, vocab_size, vocab, seed)
# with open('dicpackage.atikeep', 'wb') as handle:
#     pickle.dump(dickle, handle)
# print('dump success')
