import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data.utils import get_tokenizer

import torchtext
import datasets
import math
from tqdm import tqdm
from torchtext.vocab import build_vocab_from_iterator
# if __name__ != '__main__':
from copTrans.moduleTrans import *
# else:
# from moduleTrans import *

from spacy.lang.en.stop_words import STOP_WORDS
import spacy
import re
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
nlp = spacy.load('en_core_web_md')


def preprocessing(sentence):

    # Clear the html tag by using regular expression.
    # sentence = re.sub("<[^>]*>", "", sentence)
    # sentence = re.sub("[^\x00-\x7F]+", "", sentence) #extract non-english out
    # It matches any character which is not contained in the ASCII character set (0-127, i.e. 0x0 to 0x7F)
    stopwords = list(STOP_WORDS)
    doc = nlp(sentence)
    cleaned_tokens = []

    for token in doc:
        if token.text not in stopwords and token.pos_ != 'PUNCT' and token.pos_ != 'SPACE' and \
                token.pos_ != 'SYM' and token.pos_ != 'X':
            cleaned_tokens.append(token.lemma_.lower().strip())

    return " ".join(cleaned_tokens)


v = [line.rstrip() for line in open('vocabjjngu.txt', mode='r')]
# print('Vocab Size check', len(v)) #not work
with open('vocabjjngu.atikeep', 'rb') as handle:
    vocab = pickle.load(handle)

INPUT_DIM = len(vocab)
OUTPUT_DIM = len(vocab)
HID_DIM = 256
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1

# enc = Encoder(INPUT_DIM,
#               HID_DIM,
#               ENC_LAYERS,
#               ENC_HEADS,
#               ENC_PF_DIM,
#               ENC_DROPOUT,
#               device)

dec = Decoder(OUTPUT_DIM,
              HID_DIM,
              DEC_LAYERS,
              DEC_HEADS,
              DEC_PF_DIM,
              DEC_DROPOUT,
              device)

# SRC_PAD_IDX = PAD_IDX
# TRG_PAD_IDX = PAD_IDX

model = dec.to(device)


def generate(prompt, max_seq_len, temperature, model, tokenizer, vocab, device, seed=3407):
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    tokens = tokenizer(prompt)
    indices = [vocab[t] for t in tokens]
    batch_size = 1
    # hidden = model.init_hidden(batch_size, device)
    with torch.no_grad():
        for i in range(max_seq_len):
            src = torch.LongTensor([indices]).to(device)

            prediction, hidden = model(src, src)

            # prediction: [batch size, seq len, vocab size]
            # prediction[:, -1]: [batch size, vocab size] #probability of last vocab

            probs = torch.softmax(prediction[:, -1] / temperature, dim=-1)
            prediction = torch.multinomial(probs, num_samples=1).item()

            # if it is unk, we sample again
            while prediction == vocab['<unk>']:
                prediction = torch.multinomial(probs, num_samples=1).item()

            if prediction == vocab['<eos>']:  # if it is eos, we stop
                break
#torch.multinomial(torch.softmax(prediction[:, -1] / temperature, dim=-1) , num_samples=1).item()
            # autoregressive, thus output becomes input
            indices.append(prediction)

    itos = vocab.get_itos()
    tokens = [itos[i] for i in indices]
    return tokens
