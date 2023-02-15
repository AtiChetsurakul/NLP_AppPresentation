import pytreebank
import torch
import numpy
# from loguru import logger
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.vocab import FastText
import torch.nn as nn
import torch


class LSTM(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, output_dim, num_layers, bidirectional, dropout, pad_ix):
        super().__init__()
        # put padding_idx so asking the embedding layer to ignore padding
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_ix)
        self.lstm = nn.LSTM(emb_dim,
                            hid_dim,
                            num_layers=num_layers,
                            bidirectional=bidirectional,
                            dropout=dropout,
                            batch_first=True)
        self.fc = nn.Linear(hid_dim * 2, output_dim)

    def forward(self, text, text_lengths):
        # text = [batch size, seq len]
        embedded = self.embedding(text)

        # ++ pack sequence ++
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths.to('cpu'), enforce_sorted=False, batch_first=True)

        # embedded = [batch size, seq len, embed dim]
        packed_output, (hn, cn) = self.lstm(
            packed_embedded)  # if no h0, all zeroes

        # ++ unpack in case we need to use it ++
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True)

        # output = [batch size, seq len, hidden dim * num directions]
        # output over padding tokens are zero tensors

        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]

        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        hn = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
        # hn = [batch size, hidden dim * num directions]

        return self.fc(hn)


def inference_classification(test_list, ret_sent=False, result=[]):
    def get_set_zoro_root(sstlab): return [
        (label, line)for tree in sstlab for label, line in tree.to_labeled_lines()]

    def text_pipeline(x): return vocab(tokenizer(x))

    def yield_tokens(data_iter, tokenizer):
        for _, text in data_iter:
            yield tokenizer(text)

    def _inference_classification(test_list, ret_sent=False, result=[]):
        for sent in test_list:
            text_totorch = torch.tensor(
                text_pipeline(sent)).to(device).reshape(1, -1)
            text_length = torch.tensor(
                [text_totorch.size(1)]).to(dtype=torch.int64)
            if ret_sent:
                result.append((sent, torch.max(model(text_totorch, text_length).squeeze(
                    1).data, 1)[1].detach().cpu().numpy()))
            else:
                result.append(torch.max(model(text_totorch, text_length).squeeze(1).data, 1)[
                              1].detach().cpu().numpy())
        return result
    save_path = '/home/atichets/Desktop/JAN23/NLP_AppPresentation/Flask_webapp/pySCRPT/LSTM.pt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sst = pytreebank.load_sst()
    train_s = get_set_zoro_root(sst['train'])
    fast_vectors = FastText(language='simple')
    tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
    vocab = build_vocab_from_iterator(yield_tokens(train_s, tokenizer), specials=[
                                      '<unk>', '<pad>', '<bos>', '<eos>'])
    vocab.set_default_index(vocab["<unk>"])
    pad_ix = vocab['<pad>']
    input_dim = len(vocab)
    hid_dim = 256
    emb_dim = 300
    output_dim = 5
    num_layers = 2
    bidirectional = True
    dropout = 0.5
    model = LSTM(input_dim, emb_dim, hid_dim, output_dim,
                 num_layers, bidirectional, dropout, pad_ix).to(device)
    model.load_state_dict(torch.load(save_path))

    return _inference_classification(test_list, ret_sent, result)
    # return  model, device,tokenizer,fast_vectors,vocab


if __name__ == '__main__':

    # model, device,tokenizer,fast_vectors,vocab = long_load()
    # print('hi')
    test_case = ['The movie should have been good',
                 'What a waste of space, why are this trash even here',
                 "Ahoy, this fantastic movie all time"]
    print(inference_classification(test_case, True))
