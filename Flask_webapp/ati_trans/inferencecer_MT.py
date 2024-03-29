import torch
import pickle
if __name__ == '__main__':
    from tranF_module import *
else:
    from ati_trans.tranF_module import *
from attacut import tokenize, Tokenizer
from torchtext.data.utils import get_tokenizer

device = torch.device('cpu')

SRC_LANGUAGE, TRG_LANGUAGE = 'th', 'en'

token_transform = {}
token_transform[SRC_LANGUAGE] = Tokenizer(model="attacut-sc")
token_transform[TRG_LANGUAGE] = get_tokenizer(
    'spacy', language='en_core_web_sm')

# Define special symbols and indices
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<sos>', '<eos>']


with open('/Users/atichetsurakul/Desktop/JAN23/nlp123clone/NLP_AppPresentation/Flask_webapp/ati_trans/vocab_transform.atikeep', 'rb') as handle:
    vocab_transform = pickle.load(handle)
print(vocab_transform)


def sequential_transforms(*transforms):
    global func

    def func(txt_input):
        for transform in transforms:
            if transform == token_transform[SRC_LANGUAGE]:
                txt_input = transform.tokenize(txt_input)
            else:
                txt_input = transform(txt_input)
        return txt_input

    return func


def tensor_transform(token_ids):
    return torch.cat((torch.tensor([SOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))


text_transform = {}
for ln in [SRC_LANGUAGE, TRG_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform[ln],
                                               vocab_transform[ln],
                                               tensor_transform)


def translation(source, variants, save_path, device):
    src_text = text_transform[SRC_LANGUAGE](source).to(device)

    src_text = src_text.reshape(-1, 1)  # because batch_size is 1
    text_length = torch.tensor([src_text.size(0)]).to(dtype=torch.int64)

    input_dim = len(vocab_transform[SRC_LANGUAGE])
    output_dim = len(vocab_transform[TRG_LANGUAGE])
    emb_dim = 256
    hid_dim = 512
    dropout = 0.5
    SRC_PAD_IDX = PAD_IDX

    attn = Attention(hid_dim, variants=variants)
    enc = Encoder(input_dim,  emb_dim,  hid_dim, dropout)
    dec = Decoder(output_dim, emb_dim,  hid_dim, dropout, attn)

    model = Seq2SeqPackedAttention(enc, dec, SRC_PAD_IDX, device).to(device)

    model.load_state_dict(torch.load(save_path))
    model.eval()

    with torch.no_grad():
        output, _ = model(src_text, text_length, None, 0, True)
    output = output.squeeze(1)
    output = output[1:]
    output_max = output.argmax(1)  # returns max indices
    mapping = vocab_transform[TRG_LANGUAGE].get_itos()

    predict_setence = []
    for token in output_max:
        if mapping[token.item()] == '<eos>':
            return ' '.join(predict_setence)
        elif mapping[token.item()] == '<unk>':
            mapping[token.item()] = '___'

        predict_setence.append(mapping[token.item()])

    return ' '.join(predict_setence)


if __name__ == '__main__':
    print(translation('ฉันชอบคุณมันเผาไฟ', 'general',
          '/Users/atichetsurakul/Desktop/JAN23/nlp123clone/NLP_labsession/Hw6_MLtranslate/models/Seq2SeqPackedAttention_general.pt', device))
